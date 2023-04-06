from scipy import optimize
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import numpy as np


def bundle_adjustment(M_ini, r, o, q):
    R_ini = M_ini[0:3, 0:3]
    t_ini = M_ini[0:3, 3:4]
    rvec_ini = Rotation.from_matrix(R_ini).as_rotvec().reshape(-1, 1)
    p_ini = triangulate(M_ini, r, o, q)
    d_ini = p_ini[2:3, :].T  # distance of initial estimated points

    # # estimation
    x0 = np.concatenate([rvec_ini, t_ini, d_ini])
    # x = optimize.leastsq(eval_func, x0, args=(r,o,q), ftol=1e-03)
    x = optimize.leastsq(eval_func, x0, args=(r, o, q), ftol=1)

    rvec_est = x[0][0:3]
    t_est = x[0][3:6]
    R_est = Rotation.from_rotvec(rvec_est).as_matrix()
    M_est = np.eye(4)
    M_est[0:3, 0:3] = R_est
    M_est[0:3, 3] = t_est
    return M_est


def pose_mat2vec(M):  # [[rx],[ry],[rz],[tx],[ty],[tz]]
    R = M[0:3, 0:3]
    t = M[0:3, 3:4]
    r = Rotation.from_matrix(R).as_rotvec().reshape(-1, 1)
    return np.concatenate([r, t])


def eval_func(x, r, o, q):
    rvec = x[0:3]
    t = x[3:6]
    d = x[6:]
    A = np.eye(3)

    R = Rotation.from_rotvec(rvec).as_matrix()
    M = np.eye(4)
    M[0:3, 0:3] = R
    M[0:3, 3] = t
    Minv = np.linalg.inv(M)

    p = r * np.tile(d.reshape(1, -1), [3, 1])
    p = homogeneous_transform(Minv, p) - o
    # proj1 = ray2uv(A, p)
    # proj2 = ray2uv(A, q)
    proj1 = vec2normalized(p)
    proj2 = vec2normalized(q)

    e = (proj2 - proj1).reshape(-1)
    return e


def light_section(ray, normal):
    n = ray.shape[1]
    normals = np.tile(normal, (1, n))
    nn = np.sum(normals * normals, axis=0)
    nr = np.sum(normals * ray, axis=0)
    s = nn / nr
    S = np.tile(s, (3, 1))
    return S * ray


def M2normal(M):
    n = M[0:3, 2:3]
    invM = np.linalg.inv(M)
    invR = invM[0:3, 0:3]
    invt = invM[0:3, 3:4]
    Rn = invR @ n
    s = -invt[2, 0] / Rn[2, 0]
    return s * n


def uv_generate(A, cMw, wP, round_threshold=1):
    n = len(wP)
    UV = []
    for i in range(n):
        p = homogeneous_transform(cMw[i, :, :], wP[i])
        uv = ray2uv(A, p)
        if round_threshold > 0:
            uv = round_threshold * np.round(uv / round_threshold)
        UV.append(uv)
        # import ipdb; ipdb.set_trace()
    return UV


def M_inv(M):
    n = M.shape[0]
    invM = np.zeros((n, 4, 4))
    for i in range(n):
        invM[i, :, :] = np.linalg.inv(M[i, :, :])
    return invM


def homogeneous_transform(M, p):
    p_ = vec2homo(p)
    q_ = M @ p_
    q = homo2vec(q_)
    return q


def triangulate(M, r, o, q):  # P1 = s1 * r, P2 = s2 * q + o, P1 = M @ P2
    n = r.shape[1]
    R = M[0:3, 0:3]
    t = M[0:3, 3:4]
    P1 = np.zeros((3, n))
    for i in range(n):
        r_ = r[:, i : i + 1]
        q_ = q[:, i : i + 1]
        o_ = o[:, i : i + 1]

        A = np.hstack([r_, -R @ q_])
        b = R @ o_ + t
        A_ = A.T @ A
        b_ = A.T @ b
        S = np.linalg.inv(A_) @ b_
        # S = np.linalg.lstsq(A, b)
        P1[:, i] = S[0][0] * r[:, i]
    return P1


def homo2vec(P):  # [a, b, c, d] -> [a/d, b/c, c/d]
    n = P.shape[0]
    Q = P / np.tile(P[n - 1 : n, :], (n, 1))
    Q = Q[:-1, :]
    return Q


def vec2homo(P):  # [a, b, c] -> [a, b, c , 1]
    n = P.shape[1]
    Q = np.vstack([P, np.ones((1, n))])
    return Q


def vec2normalized(P):  # [a, b, c] -> [a, b, c]/norm([a,b,c])
    S = np.tile(np.sqrt(np.sum(P * P, axis=0)), [3, 1])
    return P / S


def ray2uv(A, P):
    p = A @ P
    p = homo2vec(p)
    return p


def uv2ray(A, uv):
    invA = np.linalg.inv(A)
    p = vec2homo(uv)
    return invA @ p


def skew(v):
    v = v.reshape(-1)
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def quadratic2linear(U, V):  # U E V = 0 -> P e = 0
    l = U.shape[1]
    m = U.shape[0]
    n = V.shape[1]
    P = np.zeros((l, m * n))

    for i in range(l):
        for j in range(m):
            for k in range(n):
                P[i, j * n + k] = U[j, i] * V[i, k]
    return P


def plot3(P):
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(P[0, :], P[1, :], P[2, :], color="r")

def three_points_algorithm(c1_P_dir, c1_t_c2):
    c2_P_dir = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
    # C1からの方向ベクトルと並進ベクトルからスケールを決定する
    a = np.dot(c1_P_dir[:, 0], c1_P_dir[:, 1])
    d = np.dot(c1_P_dir[:, 1], c1_P_dir[:, 2])
    g = np.dot(c1_P_dir[:, 2], c1_P_dir[:, 0])
    b = -np.dot(c1_P_dir[:, 0], c1_t_c2[:,0])
    e = -np.dot(c1_P_dir[:, 1], c1_t_c2[:,0])
    i = -np.dot(c1_P_dir[:, 2], c1_t_c2[:,0])
    c = e
    f = i
    h = b
    z = np.dot(c1_t_c2[:,0], c1_t_c2[:,0])

    A = a * f * i - b * d * i - c * f * g + d * g * z
    B = (
        a * f * z
        + a * i * z
        - b * e * i
        - b * d * z
        - c * g * z
        - c * f * h
        + d * h * z
        + e * g * z
    )
    C = a * z * z - b * e * z - c * h * z + e * h * z
    x_a = (-B + np.sqrt(B * B - 4 * A * C)) / (2 * A)
    x_b = (-B - np.sqrt(B * B - 4 * A * C)) / (2 * A)
    
    # 不適な解の除外
    q3 = np.array([x_a, x_b])
    q1 = -1*(i *q3+z)/(g*q3+h)
    q2 = -1*(f *q3+z)/(d*q3+e)
    c1_P_a = np.array([[q1[0], q2[0], q3[0]]]) * c1_P_dir 
    c1_P_b = np.array([[q1[1], q2[1], q3[1]]]) * c1_P_dir
    c1_P = c1_P_a if q1[0] > 0 and q2[0] > 0 and q3[0] > 0 else c1_P_b
    Q = c1_P - t
    c1_R_c2 = Q / np.linalg.norm(Q, axis=0)

    return [c1_P, c1_R_c2]