import dill
import numpy as np
import sympy as sp

from . import utils


class Laser:
    def __init__(self, origin, direction):
        if origin.shape != direction.shape:
            raise ValueError("origin and direction must have the same dimension")
        self.origin = origin
        self.direction = direction
        self.M = None
        self.P = None

    def transform(self, M):  # M[4][4]
        R = M[0:3, 0:3]
        origin = utils.homogeneous_transform(M, self.origin)
        direction = R @ self.direction.copy()
        return [origin, direction]

    def ray_trace_cylinder(self, origin, direction, params):
        radius = params["radius"]
        [dx, dy] = direction[:2, :]
        [ox, oy] = origin[:2, :]
        a = dx**2 + dy**2
        b = 2 * (dx * ox + dy * oy)
        c = ox**2 + oy**2 - radius**2

        s = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        return origin + s * direction

    def ray_trace(self, origin, direction, surface_functions):
        # 方向ベクトルと陰関数から半直線と曲面の方程式を作る
        surface_functions_eq = surface_functions.get("eq")
        surface_functions_ineq = surface_functions.get("ineq")
        ineq_logic = surface_functions.get("ineq_logic")
        s = sp.symbols("s")
        points = np.zeros((3, 0))
        for i in range(origin.shape[1]):
            x, y, z = s * direction[:, i] + origin[:, i]
            eq_list = [f(x, y, z) for f in surface_functions_eq]
            res = sp.nonlinsolve(eq_list, [s])
            s_candidates = [s for s in list(sum(res, ())) if s in sp.S.Reals and s > 0]
            for j, s_j in enumerate(s_candidates):
                x_j, y_j, z_j = s_j * direction[:, i] + origin[:, i]
                is_valid_list = [ineq(x_j, y_j, z_j) for ineq in surface_functions_ineq]
                is_valid = ineq_logic(is_valid_list)
                if is_valid:
                    point_i = s_j * direction[:, i : i + 1] + origin[:, i : i + 1]
                    points = np.hstack([points, point_i])
                    break
                if j == len(s_candidates) - 1:
                    print(f"no valid solution. origin: {origin[:, i]}, direction: {direction[:, i]}")
        return points

    def dataset_generate(self, M, surface_functions):  # M[n,4,4]
        n = len(M)
        P = []
        for i in range(n):
            [origin, direction] = self.transform(M[i])
            p = self.ray_trace(origin, direction, surface_functions)
            P.append(p)
        self.P = P
        self.M = M

    @staticmethod
    def save(laser, name):
        with open(name + ".pickle", mode="wb") as f:
            dill.dump(laser, f)

    @staticmethod
    def load(name):
        with open(name + ".pickle", mode="rb") as f:
            return dill.load(f)
