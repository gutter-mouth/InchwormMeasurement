from . import utils
import numpy as np


class Laser:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
        self.M = None
        self.P = None

    def transform(self, M):  # M[4][4]
        R = M[0:3, 0:3]
        origin = utils.homogeneous_transform(M, self.origin)
        direction = R @ self.direction
        return [origin, direction]

    def ray_trace_cylinder(self, origin, direction, params):
        radius = params["radius"]
        [dx, dy] = direction[:2, :]
        [ox, oy] = origin[:2, :]
        a = dx ** 2 + dy ** 2
        b = - dx * ox - dy * oy
        c = (dx ** 2 + dy ** 2) * radius ** 2 + 2 * dx * dy * \
            ox * oy - (dx ** 2) * (oy ** 2) - (dy ** 2) * (ox ** 2)
        s = (b + np.sqrt(c)) / a
        return origin + s * direction

    def dataset_generate(self, M, params):  # M[n,4,4]
        n = M.shape[0]
        P = []
        for i in range(n):
            [origin, direction] = self.transform(M[i, :, :])
            p = self.ray_trace_cylinder(origin, direction, params)
            P.append(p)
        self.P = P
        self.M = M
