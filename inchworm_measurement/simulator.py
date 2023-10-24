import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from sklearn.neighbors import NearestNeighbors

from . import utils

# import ipdb; ipdb.set_trace()


class Simulator:
    def __init__(self, params):
        self.A = params["A"]
        self.spot_laser = params["spot_laser"]
        self.ring_laser = params["ring_laser"]
        self.cMr = params["cMr"]
        self.idx_camera = params["idx_camera"]
        self.idx_spot = params["idx_spot"]
        self.round_threshold = params["round_threshold"]
        self.is_bundle = params["is_bundle"]
        self.is_scale_true = params["is_scale_true"]
        self.name = params["name"]

        if len(self.spot_laser.origin.T) != 3:
            raise ValueError("Points of spot laser must be three when three points algorithm is used.")

    @staticmethod
    def generate_idx(start, end, period, offset=0):
        idx_camera = []
        idx_spot = []
        for i in range(start, end, period):
            for j in range(period + 1):
                idx_camera.append(i + j)
                idx_spot.append(i + offset)
        return np.array(idx_camera), np.array(idx_spot)

    @staticmethod
    def generate_is_camera_moved(idx_camera, idx_spot):
        if len(idx_camera) != len(idx_spot):
            raise ValueError("Length of idx_camera and idx_spot must be same.")
        is_camera_moved = []
        for i in range(len(idx_camera) - 1):
            if idx_camera[i] != idx_camera[i + 1] and idx_spot[i] != idx_spot[i + 1]:
                raise ValueError("One of idx_camera and idx_spot must be same.")
            if idx_camera[i] == idx_camera[i + 1]:
                is_camera_moved.append(0)
            else:
                is_camera_moved.append(1)
        return np.array(is_camera_moved)

    @staticmethod
    def filter_used_laser(laser, idx):
        laser.P = laser.P[idx]
        laser.M = laser.M[idx]
        return laser

    @staticmethod
    def generate_2d_points(points_3d, camera_pose, A, round_threshold=0.5):
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
    def estimate_pose_whole(UV_spot, UV_spot_origin, spot_params, cP_ring, A, is_camera_moved):
        # spotレーザが同じ位置にある区間でグループ化
        n = len(is_camera_moved)
        separtor_idx = np.arange(n)[is_camera_moved == 0] + 1
        separtor_idx = [0, *separtor_idx, n + 1]
        section = []
        for current_idx, next_idx in zip(separtor_idx[:-1], separtor_idx[1:]):
            section.append(np.arange(current_idx, next_idx))

        # グループ毎にスケールs（スポットレーザ原点と0番目のスポットレーザ点間の距離）を1としたときに下記を求める
        # - spot-camera間の位置・姿勢変換
        # - spotのカメラ座標系における3次元座標
        cMs_est_dir_batch = []
        c_P_dir_batch = []
        for index_list in section:
            UV_spot_part = [UV_spot[i] for i in index_list]
            UV_spot_origin_part = [UV_spot_origin[i] for i in index_list]
            c1_P_part, _, c1_M_c2_part = Simulator.estimate_pose_batch(UV_spot_part, UV_spot_origin_part, spot_params, A)
            c_P_dir_batch.append(c1_P_part)
            cMs_est_dir_batch.append(c1_M_c2_part)

        # スケール推定
        scales = Simulator.estimate_scale(cMs_est_dir_batch, c_P_dir_batch, cP_ring)

        # 結合
        cMs_est, wMc_est, wP_spot_est = Simulator.join_batch(cMs_est_dir_batch, c_P_dir_batch, scales)
        return cMs_est, wMc_est, wP_spot_est

    @staticmethod
    def estimate_pose_batch(UV_spot, UV_spot_origin, spot_params, A):
        c1_P_batch = []
        c2_P_batch = []
        c1_M_c2_batch = []
        for UV_spot_i, UV_spot_origin_i in zip(UV_spot, UV_spot_origin):
            c_P_dir = utils.uv2ray(A, UV_spot_i)
            c_t_s_dir = utils.uv2ray(A, UV_spot_origin_i)
            [c1_P, c2_P, c1_M_c2] = utils.sfm_by_orthogonal_three_points(c_P_dir, c_t_s_dir)

            s = np.linalg.norm(c2_P[:, 0])  # 1点目のノルムが1となるように正規化
            c1_P = c1_P / s
            c2_P = c2_P / s
            c1_M_c2[0:3, 3] = c1_M_c2[0:3, 3] / s
            c1_P_batch.append(c1_P)
            c2_P_batch.append(c2_P)
            c1_M_c2_batch.append(c1_M_c2)
        # Todo spot paramsの誤差を考慮したバンドル調整によるチューニング処理を追加する
        return [c1_P_batch, c2_P_batch, c1_M_c2_batch]

    @staticmethod
    def join_batch(cMs_dir_batch, c_P_dir_batch, scales):
        cMs = []
        wMc = []
        wP = []
        wMci0 = np.eye(4)
        for i, cMs_dir_part in enumerate(cMs_dir_batch):
            ci0Ms = utils.M_scale(cMs_dir_part[0], scales[0])
            for j, cMs_dir in enumerate(cMs_dir_part):
                cijMs = utils.M_scale(cMs_dir, scales[i])
                ci0Mcij = ci0Ms @ np.linalg.inv(cijMs)
                wMcij = wMci0 @ ci0Mcij
                cMs.append(cijMs)
                wMc.append(wMcij)
                wP.append(utils.homogeneous_transform(wMcij, scales[i] * c_P_dir_batch[i][j]))
            wMci0 = wMc[-1]
        return [cMs, wMc, wP]

    @staticmethod
    def estimate_scale(cMs_dir_batch, cP_spot_dir_batch, cP_ring):
        def scale_eval_func(scales, cMs_dir_batch, cP_spot_dir_batch, cP_ring):
            [_, wMc, wP_spot] = Simulator.join_batch(cMs_dir_batch, cP_spot_dir_batch, scales)
            wP_ring = [utils.homogeneous_transform(M, P) for M, P in zip(wMc, cP_ring)]
            X = np.hstack(wP_ring).T
            Y = np.hstack(wP_spot).T
            nn = NearestNeighbors(algorithm="kd_tree", metric="euclidean", n_jobs=1)
            # Xの点群に対してインデックスを作る
            nn.fit(X)
            # YからXへの最近傍探索を行う
            distances, _ = nn.kneighbors(Y)

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

    @staticmethod
    def plot_pose(wMc_est, wMc_true):
        pose_est = np.array([utils.pose_mat2vec(M) for M in wMc_est])
        pose_true = np.array([utils.pose_mat2vec(M) for M in wMc_true])
        labels = [
            "Rotation x",
            "Rotation y",
            "Rotation z",
            "Translation x",
            "Translation y",
            "Translation z",
        ]
        for i, label in enumerate(labels):
            est_y = pose_est[:, i]
            true_y = pose_true[:, i]
            plt.scatter(range(len(est_y)), est_y, label=label + " est", s=2)
            plt.scatter(range(len(true_y)), true_y, label=label + " true", s=2)
            plt.legend()
            plt.title(label, y=-0.2)
            plt.show()

    @staticmethod
    def show_result(points_list, color_list, xlim=[], ylim=[], zlim=[], frames=[], label=""):
        # visualizer estimated points
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x", size=15, color="black")
        ax.set_ylabel("z", size=15, color="black")
        ax.set_zlabel("y", size=15, color="black")
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if zlim:
            ax.set_zlim(zlim)
        frames = frames if frames else range(len(points_list[0]))

        for [points, color] in zip(points_list, color_list):
            for i in frames:
                print(np.array(points[i]))
                print(np.isnan(points[i]).any(axis=1))
                ax.scatter(points[i][0], points[i][2], points[i][1], color=color, marker=".", s=1)
        ax.set_aspect("equal")
        ax.invert_zaxis()
        plt.show()

    def run_preprocess(self):
        spot_laser, ring_laser, idx_camera, idx_spot, cMr, A, round_threshold = (
            self.spot_laser,
            self.ring_laser,
            self.idx_camera,
            self.idx_spot,
            self.cMr,
            self.A,
            self.round_threshold,
        )

        is_camera_moved = Simulator.generate_is_camera_moved(idx_camera, idx_spot)
        wP_spot_true = [spot_laser.P[idx] for idx in idx_spot]
        wP_ring_true = [ring_laser.P[idx] for idx in idx_camera]
        wMs_true = [spot_laser.M[idx] for idx in idx_spot]
        wMc_true = [ring_laser.M[idx] @ np.linalg.inv(cMr) for idx in idx_camera]
        wP_spot_origin = [wMs[0:3, 3:4] for wMs in wMs_true]
        UV_spot_origin = Simulator.generate_2d_points(wP_spot_origin, wMc_true, A, round_threshold=round_threshold)
        UV_spot = Simulator.generate_2d_points(wP_spot_true, wMc_true, A, round_threshold=round_threshold)
        UV_ring = Simulator.generate_2d_points(wP_ring_true, wMc_true, A, round_threshold=round_threshold)
        normal = utils.M2normal(cMr)
        spot_params = {"direction": spot_laser.direction, "origin": spot_laser.origin}
        print("--- Preprocess ---")

        [
            self.UV_spot_origin,
            self.wP_spot_true,
            self.wP_ring_true,
            self.wMs_true,
            self.wMc_true,
            self.UV_spot,
            self.UV_ring,
            self.normal,
            self.spot_params,
            self.is_camera_moved,
        ] = [
            UV_spot_origin,
            wP_spot_true,
            wP_ring_true,
            wMs_true,
            wMc_true,
            UV_spot,
            UV_ring,
            normal,
            spot_params,
            is_camera_moved,
        ]

    def run_measurement(self):
        [
            A,
            UV_spot_origin,
            UV_spot,
            UV_ring,
            normal,
            spot_params,
            is_camera_moved,
        ] = [
            self.A,
            self.UV_spot_origin,
            self.UV_spot,
            self.UV_ring,
            self.normal,
            self.spot_params,
            self.is_camera_moved,
        ]

        cP_ring_est = Simulator.calculate_3dpoints(UV_ring, normal, A)
        cMs_est, wMc_est, wP_spot_est = Simulator.estimate_pose_whole(UV_spot, UV_spot_origin, spot_params, cP_ring_est, A, is_camera_moved)
        wP_ring_est = Simulator.integrate_3dpoints(cP_ring_est, wMc_est)

        [self.wMc_est, self.cMs_est, self.wP_spot_est, self.wP_ring_est, self.cP_ring_est] = [
            wMc_est,
            cMs_est,
            wP_spot_est,
            wP_ring_est,
            cP_ring_est,
        ]

    def run_show_result(self, type="estimated", label="", xlim=[], ylim=[], zlim=[], frames=[]):
        if type == "estimated":
            wP_ring = self.wP_ring_est
            wP_spot = self.wP_spot_est
        elif type == "groundtruth":
            wP_ring = self.wP_ring_true
            wP_spot = self.wP_spot_true
        elif type == "truescale":
            wP_ring = self.wP_ring_true
            wP_spot = self.wP_spot_true

        # visualizer points
        Simulator.show_result([wP_ring, wP_spot], ["green", "red"], xlim, ylim, zlim, frames, label)

    def run_evaluate(self):
        [wMc_est, wMc_true] = [self.wMc_est, self.wMc_true]
        Simulator.plot_pose(wMc_est, wMc_true)
