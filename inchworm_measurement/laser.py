from . import utils
import numpy as np
import sympy as sp
from typing import Any

Vector_3D = np.ndarray[(None,3), np.dtype[np.float64]]
Matrix_homo = np.ndarray[(None,4,4), np.dtype[np.float64]]

class Laser:
    def __init__(self, origin: Vector_3D, direction: Vector_3D):
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
    
    def ray_trace(self, origin: Vector_3D, direction: Vector_3D, surface_functions_eq:list[Any], surface_functions_ineq:list[Any])->Vector_3D:
        # 方向ベクトルと陰関数から半直線と曲面の方程式を作る
        s = sp.symbols('s', real=True)
        points = np.zeros(origin.shape)
        for i in range(origin.shape[1]):
            a, b, c = direction[:,i]
            eq_list = [f(s*a, s*b, s*c) for f in surface_functions_eq]
            ineq_list = [f(s*a, s*b, s*c) for f in surface_functions_ineq] + [s > 0]
            
            sol_candidates = list(sp.nonlinsolve(eq_list, [s]))
            for j in range(len(sol_candidates)):
                s_j = sol_candidates[j][0]
                isValid = np.all([ineq.subs(s, s_j) for ineq in ineq_list])
                if isValid:
                    points[:,i] = s_j * direction[:,i] + origin[:,i]
                    break
                else:
                    points[:,i] = [None, None, None]
        return points

    def dataset_generate(self, M: Matrix_homo, surface_functions_eq:list[Any], surface_functions_ineq:list[Any]):  # M[n,4,4]
        n = M.shape[0]
        P = []
        for i in range(n):
            print(i)
            [origin, direction] = self.transform(M[i, :, :])
            p = self.ray_trace(origin, direction, surface_functions_eq, surface_functions_ineq)
            P.append(p)
        self.P = P
        self.M = M
