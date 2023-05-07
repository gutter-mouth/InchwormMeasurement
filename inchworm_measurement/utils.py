from scipy import optimize
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import copy


def bundle_adjustment(M_ini, r, o, q):
    R_ini = M_ini[0:3, 0:3]
    t_ini = M_ini[0:3, 3:4]
    rvec_ini = Rotation.from_matrix(R_ini).as_rotvec().reshape(-1, 1)
    p_ini = triangulate(M_ini, r, o, q)
    d_ini = p_ini[2:3, :].T  # distance of initial estimated points

    # # estimation
    x0 = np.concatenate([rvec_ini, t_ini, d_ini])
    # x = optimize.leastsq(eval_func, x0, args=(r,o,q), ftol=1e-03)
    x = optimize.leastsq(eval_func, x0, args=(r, o, q), ftol=1e-03)

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


def pose_split(
    M_list, section_list
):  # [w_M_0, ...w_M_i, ...] -> [..., [j-1_M_j,..., j-1_M_i-1], [i-1_M_i, ...], ...]
    M_splited = []
    for i in range(len(section_list)):
        M_base = M_joined[section[i - 1][-1]] if i == 0 else np.eye(4)  # i-1_M_i
        M_splited.append(
            [np.linalg.inv(M_base) @ M_list[j] for j in section]
        )  # i_M_j = inv(w_M_i) @ w_M_j
    return M_splited


def pose_join(M_list_list):
    M_joined = []
    for M_list in M_list_list:
        M_base = M_joined[-1] if len(M_joined) > 0 else np.eye(4)  # w_M_i
        for M in M_list:
            M_joined.append(M_base @ M)  # w_M_j = w_M_i @ i_M_j
    return M_joined


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


def M_scale(M, s):
    M_s = copy.deepcopy(M)
    M_s[0:3, 3] = s * M[0:3, 3]
    return M_s


def homogeneous_transform(M, P):
    return homo2vec(M @ vec2homo(P))


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


def triangulate(c1_M_c2, c1_P_dir, c2_P_dir):
    n = c1_P_dir.shape[1]
    R = c1_M_c2[0:3, 0:3]
    t = c1_M_c2[0:3, 3:4]
    c1_P = np.zeros((3, n))
    c2_P = np.zeros((3, n))
    for i in range(n):
        r1 = c1_P_dir[:, i : i + 1]
        r2 = c2_P_dir[:, i : i + 1]

        A = np.hstack([r1, -R @ r2])
        b = t
        AA = A.T @ A
        bb = A.T @ b
        S = np.linalg.inv(AA) @ bb
        c1_P[:, i] = S[0, 0] * r1.flatten()
        c2_P[:, i] = S[1, 0] * r2.flatten()
    return [c1_P, c2_P]


def sfm_by_five_points(c1_P_dir, c2_P_dir):
    A = np.eye(3)

    c1_UV = ray2uv(A, c1_P_dir)
    c2_UV = ray2uv(A, c2_P_dir)

    E, mask = cv2.findEssentialMat(c1_UV.T, c2_UV.T, A)
    _, R, t, _ = cv2.recoverPose(E, c1_UV.T, c2_UV.T, A)

    c1_M_c2 = np.eye(4)
    c1_M_c2[0:3, 0:3] = R.T
    c1_M_c2[0:3, 3:4] = -R.T @ t
    [c1_P, c2_P] = triangulate(c1_M_c2, c1_P_dir, c2_P_dir)
    return [c1_P, c2_P, c1_M_c2]


def sfm_by_orthogonal_three_points(c1_P_dir, c1_t_c2):
    c2_P_dir = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
    # C1からの方向ベクトルと並進ベクトルからスケールを決定する
    a = np.dot(c1_P_dir[:, 0], c1_P_dir[:, 1])
    d = np.dot(c1_P_dir[:, 1], c1_P_dir[:, 2])
    g = np.dot(c1_P_dir[:, 2], c1_P_dir[:, 0])
    b = -np.dot(c1_P_dir[:, 0], c1_t_c2[:, 0])
    e = -np.dot(c1_P_dir[:, 1], c1_t_c2[:, 0])
    i = -np.dot(c1_P_dir[:, 2], c1_t_c2[:, 0])
    c = e
    f = i
    h = b
    z = np.dot(c1_t_c2[:, 0], c1_t_c2[:, 0])

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
    q1 = -1 * (i * q3 + z) / (g * q3 + h)
    q2 = -1 * (f * q3 + z) / (d * q3 + e)
    
    c1_P_a = np.array([[q1[0], q2[0], q3[0]]]) * c1_P_dir
    c1_P_b = np.array([[q1[1], q2[1], q3[1]]]) * c1_P_dir
    c1_P = c1_P_a if q1[0] > 0 and q2[0] > 0 and q3[0] > 0 else c1_P_b
    Q = c1_P - c1_t_c2
    c1_R_c2 = Q / np.linalg.norm(Q, axis=0)
    c1_M_c2 = np.eye(4)
    c1_M_c2[0:3, 0:3] = c1_R_c2
    c1_M_c2[0:3, 3:4] = c1_t_c2
    c2_P = homogeneous_transform(np.linalg.inv(c1_M_c2), c1_P)
    return [c1_P, c2_P, c1_M_c2]
