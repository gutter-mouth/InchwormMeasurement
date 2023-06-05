from . import utils
import numpy as np
import sympy as sp
from typing import Any
import pickle


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
        a = dx ** 2 + dy ** 2
        b = 2 * (dx * ox + dy * oy)
        c = ox**2 + oy**2 - radius ** 2

        s = (-b + np.sqrt(b**2-4*a*c)) / (2*a)
        return origin + s * direction

    def ray_trace(self, origin, direction, surface_functions):
        # 方向ベクトルと陰関数から半直線と曲面の方程式を作る
        surface_functions_eq = surface_functions.get("eq")
        surface_functions_ineq = surface_functions.get("ineq")
        ineq_logic = surface_functions.get("ineq_logic")
        s = sp.symbols('s', real=True)
        points = np.zeros(origin.shape)
        for i in range(origin.shape[1]):
            a, b, c = direction[:, i]
            eq_list = [f(s*a, s*b, s*c) for f in surface_functions_eq]
            ineq_list = [f(s*a, s*b, s*c) for f in surface_functions_ineq]

            res = sp.nonlinsolve(eq_list, [s])
            s_candidates = [s for s in list(sum(res, ())) if s > 0]

            for s_j in s_candidates:
                is_valid_list = [ineq.subs(s, s_j) for ineq in ineq_list]
                is_valid = ineq_logic(is_valid_list)
                if is_valid:
                    points[:, i] = s_j * direction[:, i] + origin[:, i]
                    break
                points[:, i] = [None, None, None]
        print(points)
        return points

    def dataset_generate(self, M, surface_functions):  # M[n,4,4]

        n = len(M)
        P = []
        for i in range(n):
            print(i)
            [origin, direction] = self.transform(M[i])
            p = self.ray_trace(origin, direction, surface_functions)
            P.append(p)
        self.P = P
        self.M = M

    @ staticmethod
    def save(laser, name):
        with open(name + ".pickle", mode="wb") as f:
            pickle.dump(laser, f)

    @ staticmethod
    def load(name):
        with open(name + ".pickle", mode="rb") as f:
            return pickle.load(f)
