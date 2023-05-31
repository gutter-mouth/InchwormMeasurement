import cv2
import numpy as np
import copy
import pickle
from sklearn.neighbors import NearestNeighbors
from scipy import optimize
from operator import itemgetter


from attrdict import AttrDict
from matplotlib import pyplot as plt
from . import utils
from .laser import Laser

# import ipdb; ipdb.set_trace()

class Simulator:
    def __init__(self, params):
        self.A = params["A"]
        self.spot_laser = params["spot_laser"]
        self.ring_laser = params["ring_laser"]
        self.cMr = params["cMr"]
        self.idx_length = params["idx_length"]
        self.idx_period = params["idx_period"]
        self.round_threshold = params["round_threshold"]
        self.is_three_points_algorithm = params["is_three_points_algorithm"]
        self.is_bundle = params["is_bundle"]
        self.is_5points_true = params["is_5points_true"]
        self.is_scale_true = params["is_scale_true"]
        self.name = params["name"]
        
        if self.is_three_points_algorithm and len(self.spot_laser.origin.T) != 3:
            raise ValueError("Points of spot laser must be three when three points algorithm is used.")
    
    @staticmethod
    def generate_idx(length, period):
        idx_camera = []
        idx_spot = []
        is_camera_moved = []
        for i in range(0, length, period):
            for j in range(period + 1):
                idx_camera.append(i + j)
                idx_spot.append(i)
        for i in range(len(idx_camera) - 1):
            if idx_camera[i] == idx_camera[i + 1]:
                is_camera_moved.append(0)
            else:
                is_camera_moved.append(1)
        return idx_camera, idx_spot, is_camera_moved
    
    @staticmethod
    def filter_used_laser(laser, idx):
        laser.P = laser.P[idx]
        laser.M = laser.M[idx]
        return laser
    
    @staticmethod    
    def generate_2d_points( points_3d, camera_pose, A, round_threshold=0.5):
        points_2d = utils.uv_generate(A, camera_pose, points_3d, round_threshold=round_threshold)
        return points_2d
    
    @staticmethod   
    def calculate_3dpoints(UV, normal, A):
        # calculate points irradiated by ring laser
        n = len(UV)
        P = []

        for i in range(n):
            ray = utils.uv2ray(A, UV[i])
            p = utils.light_section(ray, normal)
            P.append(p)
        return P
    
    @staticmethod   
    def estimate_pose(UV_spot, UV_spot_origin, spot_params, cP_ring, A, is_camera_moved, is_three_points_algorithm=True):
        section = []
        group = [0]
        for i in range(len(is_camera_moved)):
            if is_camera_moved[i]:
                group.append(i+1)
            else:
                section.append(group)
                group = [i+1]
        if group:
            section.append(group)

        cMs_est_dir_batch = []
        c_P_dir_batch = []
        for index_list in section:
            c1_P_batch, c2_P_batch, c1_M_c2_batch = Simulator.estimate_pose_batch(
                UV_spot, UV_spot_origin, spot_params, index_list, A, is_three_points_algorithm
            )
            c_P_dir_batch.append(c1_P_batch)
            cMs_est_dir_batch.append(c1_M_c2_batch)

        # スケール推定
        scales = Simulator.estimate_scale(cMs_est_dir_batch, c_P_dir_batch, cP_ring)

        # 結合
        cMs_est, wMc_est, wP_spot_est = Simulator.join_batch(cMs_est_dir_batch, c_P_dir_batch, scales)
        return cMs_est, wMc_est, wP_spot_est

    @staticmethod
    def estimate_pose_batch(UV_spot, UV_spot_origin, spot_params,index_list, A, is_three_points_algorithm):
        c1_P_batch = []
        c2_P_batch = []
        c1_M_c2_batch = []
        for i in index_list:
            c_P_dir = utils.uv2ray(A, UV_spot[i])
            s_P_dir = spot_params["direction"]
            c_t_s_dir = utils.uv2ray(A, UV_spot_origin[i])
            [c1_P, c2_P, c1_M_c2] = (
                utils.sfm_by_orthogonal_three_points(c_P_dir, c_t_s_dir)
                if is_three_points_algorithm
                else utils.sfm_by_five_points(c_P_dir, s_P_dir)
            )
            s = np.linalg.norm(c2_P[:, 0])  # 1点目のノルムが1となるように正規化
            c1_P = c1_P / s
            c2_P = c2_P / s
            c1_M_c2[0:3, 3] = c1_M_c2[0:3, 3] / s
            c1_P_batch.append(c1_P)
            c2_P_batch.append(c2_P)
            c1_M_c2_batch.append(c1_M_c2)

        return [c1_P_batch, c2_P_batch, c1_M_c2_batch]

    @staticmethod
    def join_batch(cMs_dir_batch, c_P_dir_batch, scales):
        cMs = []
        wMc = []
        wP = []
        idx = 0
        for i in range(len(cMs_dir_batch)):
            for j in range(len(cMs_dir_batch[i])):
                cMs.append(utils.M_scale(cMs_dir_batch[i][j], scales[i]))
                if len(wMc) == 0:
                    wMc.append(np.eye(4))
                elif j == 0:
                    wMc.append(wMc[idx - 1])
                else:
                    wMc.append(wMc[idx - 1] @ cMs[idx - 1] @ np.linalg.inv(cMs[idx]))
                wP.append(utils.homogeneous_transform(wMc[idx], scales[i] * c_P_dir_batch[i][j]))
                idx += 1
        return [cMs, wMc, wP]

    @staticmethod
    def estimate_scale(cMs_dir_batch, cP_spot_dir_batch, cP_ring):
        def scale_eval_func(scales, cMs_dir_batch, cP_spot_dir_batch, cP_ring):
            [cMs, wMc, wP_spot] = Simulator.join_batch(cMs_dir_batch, cP_spot_dir_batch, scales)
            wP_ring = [utils.homogeneous_transform(M, P) for M, P in zip(wMc, cP_ring)]
            X = np.hstack(wP_ring).T
            Y = np.hstack(wP_spot).T
            nn = NearestNeighbors(algorithm="kd_tree", metric="euclidean", n_jobs=1)
            # Xの点群に対してインデックスを作る
            nn.fit(X)
            # YからXへの最近傍探索を行う
            distances, indices = nn.kneighbors(Y)

            return np.array([d[0] for d in distances])

        scales_ini = np.ones((len(cMs_dir_batch)))
        result = optimize.least_squares(
            scale_eval_func,
            scales_ini,
            method="lm",
            args=(cMs_dir_batch, cP_spot_dir_batch, cP_ring),
        )
        scales = result["x"]
        return scales
    
    @staticmethod
    def integrate_3dpoints(cP_ring_est, wMc_est):
        # calculate points irradiated by ring laser
        n = len(cP_ring_est)
        wP_ring_est = []
        
        for i in range(n):
            cp = cP_ring_est[i]
            wp = utils.homogeneous_transform(wMc_est[i], cp)
            wP_ring_est.append(wp)
        return wP_ring_est

    def calc_error(self):
        # evaluation
        n = len(self.wP_ring_est)
        error = 0
        for i in range(n):
            dp = self.wP_ring_est[i] - self.wP_ring_true[i]
            m = dp.shape[1]
            ei = 0
            for j in range(m):
                dpj = dp[:, j]
                ej = np.sqrt(np.sum(dpj**2))
                ei += ej
            ei /= m
            error += ei
        error /= n
        self.error = error

    def calc_error_pose_adjacent(self):
        # evaluation of pose error
        n = len(self.wP_ring_est)
        error = []
        for i in range(n - 1):
            Pose_true = np.linalg.inv(self.wMc_true[i]) @ self.wMc_true[i + 1]  # c_i+1 M c_i
            Pose_est = np.linalg.inv(self.wMc_est[i]) @ self.wMc_est[i + 1]
            pose_error = utils.pose_mat2vec(Pose_est) - utils.pose_mat2vec(Pose_true)
            error.append(pose_error)
        self.error_pose_adjacent = np.array(error).T.reshape((6, -1))

    def calc_error_pose_global(self):
        # evaluation of pose error
        n = len(self.wP_ring_est)
        error = []
        for i in range(n):
            Pose_true = self.wMc_true[i]
            Pose_est = self.wMc_est[i]
            pose_error = utils.pose_mat2vec(Pose_est) - utils.pose_mat2vec(Pose_true)
            error.append(pose_error)

        self.error_pose_global = np.array(error).T.reshape((6, -1))

    def show_error_pose_adjacent(self):
        e = self.error_pose_adjacent

        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        ax1.set_xlabel("index")
        ax1.set_ylabel("Rotation Error [radian]")
        ax1.scatter(e[0, :], color="r")
        ax1.scatter(e[1, :], color="g")
        ax1.scatter(e[2, :], color="b")

        ax2.set_xlabel("index")
        ax2.set_ylabel("Translation Error [mm]")
        ax2.scatter(e[3, :], color="r")
        ax2.scatter(e[4, :], color="g")
        ax2.scatter(e[5, :], color="b")

    def show_error_pose_global(self):
        e = self.error_pose_global
        n = e.shape[1]

        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        ax1.set_xlabel("index")
        ax1.set_ylabel("Rotation Error [radian]")
        ax1.scatter(range(n), e[0, :], color="r")
        ax1.scatter(range(n), e[1, :], color="g")
        ax1.scatter(range(n), e[2, :], color="b")

        ax2.set_xlabel("index")
        ax2.set_ylabel("Translation Error [mm]")
        ax2.scatter(range(n), e[3, :], color="r")
        ax2.scatter(range(n), e[4, :], color="g")
        ax2.scatter(range(n), e[5, :], color="b")

    def show_result(self, xlim=[], ylim=[], zlim=[], frames=[], is_groundtruth=False, save_name=""):
        # visualizer estimated points
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x", size=15, color="black")
        ax.set_ylabel("y", size=15, color="black")
        ax.set_zlabel("z", size=15, color="black")
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if zlim:
            ax.set_zlim(zlim)
            
        
        label = "Result of ground truth" if is_groundtruth else "Result of estimation"
        ax.set_title(label,  y=-0.2)

        for i in frames if frames else range(len(self.wP_ring_est)):
            p = self.wP_ring_true[i] if is_groundtruth else self.wP_ring_est[i]
            q = self.wP_spot_true[i] if is_groundtruth else self.wP_spot_est[i]
            ax.scatter(p[0, :], p[1, :], p[2, :], color="g", marker=".", s=5)
            ax.scatter(q[0, :], q[1, :], q[2, :], color="r", marker="x", s=5)


        if save_name != "":
            plt.savefig(save_name, dpi=120)
        plt.show()

    def show_module(self, frames):
        # show modules in frames
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlabel("x", size=15, color="black")
        ax.set_ylabel("y", size=15, color="black")
        ax.set_zlabel("z", size=15, color="black")

        for f in frames:
            # camera axis
            T = self.wMc_est[f]
            rx = T[0:3, 0:1]
            ry = T[0:3, 1:2]
            rz = T[0:3, 2:3]
            o = T[0:3, 3:4]
            axis_x = np.hstack([o, o + rx])
            axis_y = np.hstack([o, o + ry])
            axis_z = np.hstack([o, o + rz])
            ax.plot(axis_x[0, :], axis_x[1, :], axis_x[2, :], color="r")
            ax.plot(axis_y[0, :], axis_y[1, :], axis_y[2, :], color="g")
            ax.plot(axis_z[0, :], axis_z[1, :], axis_z[2, :], color="b")

            # spot laser axis
            S = self.wMc_est[f] @ self.cMs_est[f]
            rx = S[0:3, 0:1]
            ry = S[0:3, 1:2]
            rz = S[0:3, 2:3]
            o = S[0:3, 3:4]
            axis_x = np.hstack([o, o + rx])
            axis_y = np.hstack([o, o + ry])
            axis_z = np.hstack([o, o + rz])
            ax.plot(axis_x[0, :], axis_x[1, :], axis_x[2, :], color="r")
            ax.plot(axis_y[0, :], axis_y[1, :], axis_y[2, :], color="g")
            ax.plot(axis_z[0, :], axis_z[1, :], axis_z[2, :], color="b")
        # plt.savefig("3Dpathpatch.jpg", dpi=120)
        plt.show()
    
    def plot_pose(self):
        pose_est = np.array([utils.pose_mat2vec(M) for M in self.wMc_est])
        pose_true = np.array([utils.pose_mat2vec(M) for M in self.wMc_true])
        labels = ["Rotation x", "Rotation y", "Rotation z", "Translation x", "Translation y", "Translation z"]
        for i, label in enumerate(labels):
            est_y = pose_est[:, i]
            true_y = pose_true[:, i]
            plt.scatter(range(len(est_y)), est_y, label=label+" est", s=2)
            plt.scatter(range(len(true_y)), true_y, label=label+" true", s=2)
            plt.legend()        
            plt.title(label,  y=-0.2)
            plt.show()
        # グラフを表示する
    
    def run_preprocess(self):
        spot_laser, ring_laser, idx_length, idx_period, cMr , A= self.spot_laser, self.ring_laser, self.idx_length, self.idx_period, self.cMr, self.A
        
        idx_camera, idx_spot, is_camera_moved = Simulator.generate_idx(idx_length, idx_period)
        wP_spot_true = [spot_laser.P[idx] for idx in idx_spot]
        wP_ring_true = [ring_laser.P[idx] for idx in idx_camera]
        wMs_true = [spot_laser.M[idx] for idx in idx_spot]
        wMc_true = [ring_laser.M[idx] @ np.linalg.inv(cMr) for idx in idx_camera]
        wP_spot_origin = [wMs[0:3,3:4] for wMs in wMs_true]
        UV_spot_origin = Simulator.generate_2d_points(wP_spot_origin, wMc_true, A)
        UV_spot = Simulator.generate_2d_points(wP_spot_true, wMc_true, A)
        UV_ring = Simulator.generate_2d_points(wP_ring_true, wMc_true, A)
        normal = utils.M2normal(cMr)
        spot_params = {"direction": spot_laser.direction, "origin": spot_laser.origin}

        [self.UV_spot_origin, self.wP_spot_true, self.wP_ring_true, self.wMs_true, self.wMc_true, self.UV_spot, self.UV_ring, self.normal, self.spot_params, self.is_camera_moved] = [UV_spot_origin, wP_spot_true, wP_ring_true, wMs_true, wMc_true, UV_spot, UV_ring, normal, spot_params,is_camera_moved]

    def run_measurement(self):
        [A, is_three_points_algorithm, UV_spot_origin, wP_spot_true, wP_ring_true, wMs_true, wMc_true, UV_spot, UV_ring, normal, spot_params, is_camera_moved] = [self.A, self.is_three_points_algorithm, self.UV_spot_origin, self.wP_spot_true, self.wP_ring_true, self.wMs_true, self.wMc_true, self.UV_spot, self.UV_ring, self.normal, self.spot_params, self.is_camera_moved]
        
        cP_ring_est = Simulator.calculate_3dpoints(UV_ring, normal, A)
        cMs_est, wMc_est, wP_spot_est =  Simulator.estimate_pose(UV_spot, UV_spot_origin, spot_params, cP_ring_est, A, is_camera_moved, is_three_points_algorithm)
        print(len(wMc_true), len(cP_ring_est))
        wP_ring_est = Simulator.integrate_3dpoints(cP_ring_est, wMc_est)
        
        [self.wMc_est, self.cMs_est, self.wP_spot_est, self.wP_ring_est, self.cP_ring_est] = [wMc_est, cMs_est, wP_spot_est, wP_ring_est, cP_ring_est]


        
    ##########################################

    # def set_rendered_points(self):
    #     self.spot_laser.dataset_generate(self.wMs, self.surface_functions_eq, self.surface_functions_ineq)
    #     self.ring_laser.dataset_generate(self.wMr, self.surface_functions_eq, self.surface_functions_ineq)
        

    # def set_base_motion(self):
    #     # set pose of camera, spot laser, and ring laser
    #     if not (self.base_motion[0,:,:]==np.eye(4)).all():
    #         raise ValueError("First pose of base motion must be identity matrix.")
    #     m = len(self.base_motion)
    #     wMc = self.base_motion
    #     wMs = np.array([wMc[i, :, :] @ self.spotlaser_offset for i in range(m)])
    #     wMr = np.array([wMc[i, :, :] @ self.ringlaser_offset for i in range(m)])

    #     if self.is_ring_with_camera:
    #         xMr = np.linalg.inv(wMc[0, :, :]) @ wMr[0, :, :]
    #     else:
    #         xMr = np.linalg.inv(wMs[0, :, :]) @ wMr[0, :, :]

    #     self.wMc = wMc
    #     self.wMs = wMs
    #     self.wMr = wMr
    #     self.xMr = xMr

    # def set_used_index(self):
    #     # set way of motion as a whole
    #     length = self.idx_length
    #     period = self.idx_period
    #     idx_c = []
    #     idx_s = []

    #     for i in range(0, length, period):
    #         for j in range(period + 1):
    #             idx_c.append(i + j)
    #             idx_s.append(i)

    #     is_camera_moved = []
    #     for i in range(len(idx_c) - 1):
    #         if idx_c[i] == idx_c[i + 1]:
    #             is_camera_moved.append(0)
    #         else:
    #             is_camera_moved.append(1)
    #     if self.is_ring_with_camera:  # ring laser fixed to camera
    #         idx_r = idx_c
    #     else:  # ring laser fixed to spot laser
    #         idx_r = idx_s
    #     self.idx_c = idx_c
    #     self.idx_s = idx_s
    #     self.idx_r = idx_r
    #     self.is_camera_moved = is_camera_moved

    # def select_used_M(self):
    #     n = len(self.wP_ring_est)
    #     wMc_idx = self.wMc[self.idx_c, :, :]
    #     wMs_idx = self.wMs[self.idx_s, :, :]
    #     wMr_idx = self.wMr[self.idx_r, :, :]
    #     cMw_idx = utils.M_inv(wMc_idx)
    #     cMs_idx = np.zeros((n, 4, 4))
    #     for i in range(n):
    #         cMs_idx[i, :, :] = cMw_idx[i, :, :] @ wMs_idx[i, :, :]

    #     self.wMc_true = wMc_idx
    #     self.wMs_idx = wMs_idx
    #     self.wMr_idx = wMr_idx
    #     self.cMw_idx = cMw_idx
    #     self.cMs_idx = cMs_idx

    # def generate_2dpoints(self):  # pick up indexed M, P
    #     wP_lsp_idx = [self.spot_laser.P[idx] for idx in self.idx_s]
    #     wP_r_idx = [self.ring_laser.P[idx] for idx in self.idx_r]

    #     UV_lsp_idx = utils.uv_generate(
    #         self.A, self.cMw_idx, wP_lsp_idx, round_threshold=self.round_threshold
    #     )
    #     UV_r_idx = utils.uv_generate(
    #         self.A, self.cMw_idx, wP_r_idx, round_threshold=self.round_threshold
    #     )

    #     self.wP_lsp_idx = wP_lsp_idx
    #     self.wP_r_idx = wP_r_idx
    #     self.UV_lsp_idx = UV_lsp_idx
    #     self.UV_r_idx = UV_r_idx

    # def estimate_pose(self):
    #     section = []
    #     group = []
    #     for i in range(len(self.wP_ring_est)):
    #         if i != 0 and not self.is_camera_moved[i - 1]:
    #             section.append(group)
    #             group = []
    #         group.append(i)
    #     if group:
    #         section.append(group)

    #     cMs_est_dir_batch = []
    #     c_P_dir_batch = []
    #     for index_list in section:
    #         c1_P_batch, c2_P_batch, c1_M_c2_batch = self.estimate_pose_batch(
    #             index_list, self.is_three_points_algorithm
    #         )
    #         c_P_dir_batch.append(c1_P_batch)
    #         cMs_est_dir_batch.append(c1_M_c2_batch)

    #     # スケール推定
    #     scales = self.estimate_scale(cMs_est_dir_batch, c_P_dir_batch, self.cP_r_est)

    #     # 結合
    #     [cMs_est, wMc_est, wP_spot_est] = self.join_batch(cMs_est_dir_batch, c_P_dir_batch, scales)
    #     self.cMs_est = cMs_est
    #     self.wMc_est = wMc_est
    #     self.wP_spot_est = wP_spot_est

    # def estimate_pose_batch(self, index_list, is_three_points_algorithm):
    #     c1_P_batch = []
    #     c2_P_batch = []
    #     c1_M_c2_batch = []
    #     for i in index_list:
    #         c_P_dir = utils.uv2ray(self.A, self.UV_lsp_idx[i])
    #         s_P_dir = self.spot_laser.direction
    #         c_t_s_dir = self.cMs_idx[i][0:3, 3:4]
    #         [c1_P, c2_P, c1_M_c2] = (
    #             utils.sfm_by_orthogonal_three_points(c_P_dir, c_t_s_dir)
    #             if is_three_points_algorithm
    #             else utils.sfm_by_five_points(c_P_dir, s_P_dir)
    #         )
    #         s = np.linalg.norm(c2_P[:, 0])  # 1点目のノルムが1となるように正規化
    #         c1_P = c1_P / s
    #         c2_P = c2_P / s
    #         c1_M_c2[0:3, 3] = c1_M_c2[0:3, 3] / s
    #         c1_P_batch.append(c1_P)
    #         c2_P_batch.append(c2_P)
    #         c1_M_c2_batch.append(c1_M_c2)

    #     return [c1_P_batch, c2_P_batch, c1_M_c2_batch]

    # def join_batch(self, cMs_dir_batch, c_P_dir_batch, scales):
    #     cMs = []
    #     wMc = []
    #     wP = []
    #     idx = 0
    #     for i in range(len(cMs_dir_batch)):
    #         for j in range(len(cMs_dir_batch[i])):
    #             cMs.append(utils.M_scale(cMs_dir_batch[i][j], scales[i]))
    #             if len(wMc) == 0:
    #                 wMc.append(np.eye(4))
    #             elif j == 0:
    #                 wMc.append(wMc[idx - 1])
    #             else:
    #                 wMc.append(wMc[idx - 1] @ cMs[idx - 1] @ np.linalg.inv(cMs[idx]))
    #             wP.append(utils.homogeneous_transform(wMc[idx], scales[i] * c_P_dir_batch[i][j]))
    #             idx += 1
    #     return [cMs, wMc, wP]

    # def estimate_scale(self, cMs_dir_batch, cP_spot_dir_batch, cP_ring):
    #     def scale_eval_func(scales, cMs_dir_batch, cP_spot_dir_batch, cP_ring):
    #         [cMs, wMc, wP_spot] = self.join_batch(cMs_dir_batch, cP_spot_dir_batch, scales)
    #         wP_ring = [utils.homogeneous_transform(M, P) for M, P in zip(wMc, cP_ring)]
    #         X = np.hstack(wP_ring).T
    #         Y = np.hstack(wP_spot).T
    #         nn = NearestNeighbors(algorithm="kd_tree", metric="euclidean", n_jobs=1)
    #         # Xの点群に対してインデックスを作る
    #         nn.fit(X)
    #         # YからXへの最近傍探索を行う
    #         distances, indices = nn.kneighbors(Y)
    #         # print(np.array([d[0] for d in distances]).sum())
    #         return np.array([d[0] for d in distances])

    #     scales_ini = np.ones((len(cMs_dir_batch)))
    #     result = optimize.least_squares(
    #         scale_eval_func,
    #         scales_ini,
    #         method="lm",
    #         args=(cMs_dir_batch, cP_spot_dir_batch, cP_ring),
    #     )
    #     scales = result["x"]
    #     return scales

    # def set_ring_normals(self):
    #     # calculate normal of ring laser
    #     n = len(self.wP_ring_est)
    #     cMr_est = np.zeros((n, 4, 4))
    #     normals = np.zeros((3, n))

    #     for i in range(n):
    #         if self.is_ring_with_camera:
    #             cMr_i = self.xMr
    #         else:
    #             cMr_i = self.cMs_est[i, :, :] @ self.xMr
    #         cMr_est[i, :, :] = cMr_i
    #         normals[:, i] = utils.M2normal(cMr_i).reshape(-1)

    #     self.cMr_est = cMr_est
    #     self.normals = normals





    # def run(self):
    #     self.set_base_motion()
    #     self.set_rendered_points()
    #     self.set_used_index()
    #     self.select_used_M()
    #     self.set_ring_normals()
    #     self.generate_2dpoints()
    #     self.calculate_3dpoints()
    #     self.estimate_pose()
    #     self.integrate_3dpoints()
    #     self.calc_error()
