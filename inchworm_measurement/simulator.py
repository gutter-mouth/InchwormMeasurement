from distutils.log import error
import cv2
import numpy as np

from attrdict import AttrDict
from matplotlib import pyplot as plt
from . import utils
from .laser import Laser

# import ipdb; ipdb.set_trace()


class Simulator:
    def __init__(self):

        params = {
            "A": np.eye(3),
            "LSP": self.get_default_laser_spot1(),
            "LSS": self.get_default_laser_spot2(),
            "LR": self.get_default_laser_ring(),
            "cylinder_radius": 2500,
            "base_motion": np.array(
                [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 10 * i], [0, 0, 0, 1]] for i in range(100)]
            ),
            "spotlaser_offset": np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1000], [0, 0, 0, 1]]
            ),
            "ringlaser_offset": np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1500], [0, 0, 0, 1]]
            ),
            "is_ring_with_camera": True,
            "idx_length": 100,
            "idx_period": 5,
            "round_threshold": 0.00001,
            "is_three_points": True,
            "is_bundle": False,
            "is_5points_true": False,
            "is_scale_true": False,
        }
        self.set_params(**params)

    def set_params(self, **params):
        for key in params:
            setattr(self, key, params[key])

    def get_idx(length, period):
        camera_idx = []
        spot_idx = []
        for i in length:
            for j in period:
                camera_idx.append(i + j)
                spot_idx.append(i)

    def get_default_laser_spot1(self):
        # spot laser for posture estimation
        n = 6
        origin1 = np.zeros((3, n))
        direction1 = np.array(
            [[np.cos(i * np.pi / n * 2), np.sin(i * np.pi / n * 2), 1] for i in range(n)]
        ).T
        LSP = Laser(origin1, direction1)
        return LSP

    def get_default_laser_spot2(self):
        # spot laser for scale estimation
        origin2 = np.array([[1000, 0, 0]]).T
        direction2 = np.array([[0, 1, 0]]).T
        LSS = Laser(origin2, direction2)
        return LSS

    def get_default_laser_ring(self):
        # ring laser for structured light
        m = 100
        origin3 = np.zeros((3, m))
        direction3 = np.array(
            [[np.cos(i * np.pi / m * 2), np.sin(i * np.pi / m * 2), 0] for i in range(m)]
        ).T
        LR = Laser(origin3, direction3)
        return LR

    def set_rendered_points(self):
        params = {"radius": self.cylinder_radius}
        self.LSP.dataset_generate(self.wMs, params)
        self.LSS.dataset_generate(self.wMs, params)
        self.LR.dataset_generate(self.wMr, params)

    def set_base_motion(self):
        # set pose of camera, spot laser, and ring laser
        m = len(self.base_motion)
        wMc = self.base_motion
        wMs = np.array([wMc[i, :, :] @ self.spotlaser_offset for i in range(m)])
        wMr = np.array([wMc[i, :, :] @ self.ringlaser_offset for i in range(m)])

        if self.is_ring_with_camera:
            xMr = np.linalg.inv(wMc[0, :, :]) @ wMr[0, :, :]
        else:
            xMr = np.linalg.inv(wMs[0, :, :]) @ wMr[0, :, :]

        self.wMc = wMc
        self.wMs = wMs
        self.wMr = wMr
        self.xMr = xMr

    def set_used_index(self):
        # set way of motion as a whole
        length = self.idx_length
        period = self.idx_period
        idx_c = []
        idx_s = []

        for i in range(0, length, period):
            for j in range(period + 1):
                idx_c.append(i + j)
                idx_s.append(i)

        is_camera_moved = []
        for i in range(len(idx_c) - 1):
            if idx_c[i] == idx_c[i + 1]:
                is_camera_moved.append(0)
            else:
                is_camera_moved.append(1)
        if self.is_ring_with_camera:  # ring laser fixed to camera
            idx_r = idx_c
        else:  # ring laser fixed to spot laser
            idx_r = idx_s
        self.idx_c = idx_c
        self.idx_s = idx_s
        self.idx_r = idx_r
        self.is_camera_moved = is_camera_moved

    def select_used_M(self):
        n = len(self.idx_c)
        wMc_idx = self.wMc[self.idx_c, :, :]
        wMs_idx = self.wMs[self.idx_s, :, :]
        wMr_idx = self.wMr[self.idx_r, :, :]
        cMw_idx = utils.M_inv(wMc_idx)
        cMs_idx = np.zeros((n, 4, 4))
        for i in range(n):
            cMs_idx[i, :, :] = cMw_idx[i, :, :] @ wMs_idx[i, :, :]

        self.wMc_idx = wMc_idx
        self.wMs_idx = wMs_idx
        self.wMr_idx = wMr_idx
        self.cMw_idx = cMw_idx
        self.cMs_idx = cMs_idx

    def generate_2dpoints(self):  # pick up indexed M, P
        wP_lsp_idx = [self.LSP.P[idx] for idx in self.idx_s]
        wP_lss_idx = [self.LSS.P[idx] for idx in self.idx_s]
        wP_r_idx = [self.LR.P[idx] for idx in self.idx_r]

        UV_lsp_idx = utils.uv_generate(
            self.A, self.cMw_idx, wP_lsp_idx, round_threshold=self.round_threshold
        )
        UV_lss_idx = utils.uv_generate(
            self.A, self.cMw_idx, wP_lss_idx, round_threshold=self.round_threshold
        )
        UV_r_idx = utils.uv_generate(
            self.A, self.cMw_idx, wP_r_idx, round_threshold=self.round_threshold
        )

        self.wP_lsp_idx = wP_lsp_idx
        self.wP_lss_idx = wP_lss_idx
        self.wP_r_idx = wP_r_idx
        self.UV_lsp_idx = UV_lsp_idx
        self.UV_lss_idx = UV_lss_idx
        self.UV_r_idx = UV_r_idx

    def estimate_pose(self):
        section = []
        group = []
        for i, is_moved in enumerate(self.is_camera_moved):
            if is_moved == 1:
                group.append(i)
            elif group:
                section.append(group)
                group = []
        if group:
            section.append(group)

        cMs_est_dir_batch = []
        s_P_dir_batch = []
        for index_list in section:
            cMs_est_dir, s_P_dir = self.estimate_pose_batch(index_list)

    def estimate_pose_batch(self, index_list):
        c1_P_batch = []
        c2_P_batch = []
        c1_M_c2_batch = []
        for k in range(i, j):
            c_P_dir = utils.uv2ray(self.A, self.UV_lsp_idx[i])
            s_P_dir = self.LSP.direction
            [c1_P, c2_P, c1_M_c2] = utils.sfm_by_five_points(c_P_dir, s_P_dir)
            s = c2_P[2, 0]  # 1点目のz座標が1となるように正規化
            c1_P = c1_P / s
            c2_P = c2_P / s
            c1_M_c2[0:3, 3] = c1_M_c2[0:3, 3] / s
            c1_P_batch.append(c1_P / s)
            c2_P_batch.append(c2_P / s)
            c1_M_c2_batch.append(c1_P / s)

        return [c1_P_batch, c2_P_batch, c1_M_c2_batch]

    def estimate_pose_(self):
        # estimate pose between camera and spot laser at the same time
        section = []
        group = []
        for i, is_moved in enumerate(self.is_camera_moved):
            if is_moved == 1:
                group.append(i)
            elif group:
                section.append(group)
                group = []
        if group:
            section.append(group)

        camera_stopped_index = [i for i, x in enumerate(self.is_camera_moved) if x == 0]
        n = len(self.idx_c)
        cMs_est = np.zeros((n, 4, 4))
        cMs_ini = np.zeros((n, 4, 4))

        for i in range(n):
            # r = homogeneous_transform(cMw_idx[i], wP_s_idx[i])
            [cMs_est[i, :, :], cMs_ini[i, :, :]] = self.estimate_pose_batch(i)

        # connect cMs according to is_camera_moved
        wMc_est = np.zeros((n, 4, 4))
        wMc_est[0, :, :] = np.eye(4, 4)
        for i in range(n - 1):
            if self.is_camera_moved[i]:
                # c_i_M_c_i+1
                M_rel = cMs_est[i, :, :] @ np.linalg.inv(cMs_est[i + 1, :, :])
                wMc_est[i + 1, :, :] = wMc_est[i, :, :] @ M_rel
            else:
                wMc_est[i + 1, :, :] = wMc_est[i, :, :]
        self.wMc_est = wMc_est
        self.cMs_est = cMs_est

    # def estimate_pose_i(self, i):
    #     # spot laser for posture estimation
    #     r1 = utils.uv2ray(self.A, self.UV_lsp_idx[i])
    #     # spot laser for scale estimation
    #     r2 = utils.uv2ray(self.A, self.UV_lss_idx[i])
    #     q1 = self.LSP.direction
    #     q2 = self.LSS.direction
    #     o1 = self.LSP.origin
    #     o2 = self.LSS.origin

    #     r1_ = self.UV_lsp_idx[i]
    #     q1_ = utils.ray2uv(self.A, q1)

    #     E, mask = cv2.findEssentialMat(q1_.T, r1_.T, self.A)

    #     if self.is_5points_true:
    #         R_ini = self.cMs_idx[i, 0:3, 0:3]
    #         t_ini = self.cMs_idx[i, 0:3, 3:4]
    #     else:
    #         points, R_ini, t_ini, mask_pose = cv2.recoverPose(E, q1_.T, r1_.T, self.A)

    #     # t_ini = np.sqrt(np.sum(t * t)) * t_ini.reshape(-1)
    #     if self.is_scale_true:
    #         s_ini = np.linalg.norm(t_ini, ord=2)
    #         s_true = np.linalg.norm(self.cMs_idx[i, 0:3, 3:4], ord=2)
    #         t_ini = s_true / s_ini * t_ini
    #     else:
    #         c = np.cross(r2[:, 0], R_ini @ q2[:, 0])  # todo spot2が複数要素の場合に対応
    #         s = -(c @ (R_ini @ o2[:, 0])) / (c @ t_ini[:, 0])
    #         t_ini = s * t_ini

    #     T_ini = np.eye(4)
    #     T_ini[0:3, 0:3] = R_ini
    #     T_ini[0:3, 3:4] = t_ini

    #     if self.is_bundle:
    #         r = np.hstack([r1, r2])
    #         o = np.hstack([o1, o2])
    #         q = np.hstack([q1, q2])
    #         T_est = utils.bundle_adjustment(T_ini, r, o, q)
    #     else:
    #         T_est = T_ini

    #     return [T_est, T_ini]

    def set_ring_normals(self):
        # calculate normal of ring laser
        n = len(self.idx_c)
        cMr_est = np.zeros((n, 4, 4))
        normals = np.zeros((3, n))

        for i in range(n):
            if self.is_ring_with_camera:
                cMr_i = self.xMr
            else:
                cMr_i = self.cMs_est[i, :, :] @ self.xMr
            cMr_est[i, :, :] = cMr_i
            normals[:, i] = utils.M2normal(cMr_i).reshape(-1)

        self.cMr_est = cMr_est
        self.normals = normals

    def calculate_3dpoints(self):
        # calculate points irradiated by ring laser
        n = len(self.idx_c)
        wP_r_est = []
        cP_r_est = []
        wP_r_true = []

        for i in range(n):
            ray = utils.uv2ray(self.A, self.UV_r_idx[i])
            normal = self.normals[:, i : i + 1]
            cp = utils.light_section(ray, normal)
            wp = utils.homogeneous_transform(self.wMc_est[i, :, :], cp)
            cP_r_est.append(wp)
            wP_r_est.append(wp)
            wP_r_true.append(self.LR.P[self.idx_c[i]])  # ground truth

        self.wP_r_est = wP_r_est
        self.cP_r_est = cP_r_est
        self.wP_r_true = wP_r_true

    def calc_error(self):
        # evaluation
        n = len(self.idx_c)
        error = 0
        for i in range(n):
            dp = self.wP_r_est[i] - self.wP_r_true[i]
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
        n = len(self.idx_c)
        error = []
        for i in range(n - 1):
            Pose_true = np.linalg.inv(self.wMc_idx[i]) @ self.wMc_idx[i + 1]  # c_i+1 M c_i
            Pose_est = np.linalg.inv(self.wMc_est[i]) @ self.wMc_est[i + 1]
            pose_error = utils.pose_mat2vec(Pose_est) - utils.pose_mat2vec(Pose_true)
            error.append(pose_error)
        self.error_pose_adjacent = np.array(error).T.reshape((6, -1))

    def calc_error_pose_global(self):
        # evaluation of pose error
        n = len(self.idx_c)
        error = []
        for i in range(n):
            Pose_true = self.wMc_idx[i]
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

    def show_result(self, save_name=""):
        # visualizer estimated points
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlabel("x", size=15, color="black")
        ax.set_ylabel("y", size=15, color="black")
        ax.set_zlabel("z", size=15, color="black")

        for p in self.wP_r_est:
            ax.scatter(p[0, :], p[1, :], p[2, :], color="g", marker=".", s=5)

        if save_name != "":
            plt.savefig(save_name, dpi=120)
        plt.show()

    def show_groundtruth(self):
        # visualizer estimated points
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlabel("x", size=15, color="black")
        ax.set_ylabel("y", size=15, color="black")
        ax.set_zlabel("z", size=15, color="black")

        for p in self.wP_r_true:
            ax.scatter(p[0, :], p[1, :], p[2, :], color="g", marker=".", s=5)

        # plt.savefig("3Dpathpatch.jpg", dpi=120)
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

    def run(self):
        self.set_base_motion()
        self.set_rendered_points()
        self.set_used_index()
        self.select_used_M()
        self.generate_2dpoints()
        self.estimate_pose()
        self.set_ring_normals()
        self.calculate_3dpoints()
