import pretty_errors
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


def pose_M2Rt(M):
    R = M[0:3, 0:3]
    t = M[0:3, 3:4]
    return [R, t]


def pose_Rt2M(R, t):
    M = np.eye(4)
    M[0:3, 0:3] = R
    M[0:3, 3:4] = t
    return M


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


def uv_generate(A, wMc, wP, round_threshold=1):
    n = len(wP)
    cMw = np.linalg.inv(wMc)
    UV = []
    for i in range(n):
        p = homogeneous_transform(cMw[i], wP[i])
        uv = ray2uv(A, p)
        if round_threshold > 0 and p.shape[1] > 2:
            uv = uv.astype(np.float64)
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
    Q = P / np.tile(P[n - 1: n, :], (n, 1))
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
    print(A, P)
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


def vect2col(v):
    return np.array(v).reshape(-1, 1)


def vect2row(v):
    return np.array(v).reshape(1, -1)


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
        r1 = c1_P_dir[:, i: i + 1]
        r2 = c2_P_dir[:, i: i + 1]

        A = np.hstack([r1, -R @ r2])
        b = t
        AA = A.T @ A
        bb = A.T @ b
        S = np.linalg.inv(AA) @ bb
        c1_P[:, i] = S[0, 0] * r1.flatten()
        c2_P[:, i] = S[1, 0] * r2.flatten()
    return [c1_P, c2_P]


# def sfm_by_five_points(c1_P_dir, c2_P_dir):
#     A = np.eye(3)

#     c1_UV = ray2uv(A, c1_P_dir)
#     c2_UV = ray2uv(A, c2_P_dir)

#     E, mask = cv2.findEssentialMat(c1_UV.T, c2_UV.T, A)
#     _, R, t, _ = cv2.recoverPose(E, c1_UV.T, c2_UV.T, A)

#     c1_M_c2 = np.eye(4)
#     c1_M_c2[0:3, 0:3] = R.T
#     c1_M_c2[0:3, 3:4] = -R.T @ t
#     [c1_P, c2_P] = triangulate(c1_M_c2, c1_P_dir, c2_P_dir)
#     return [c1_P, c2_P, c1_M_c2]


def plot_pose(M_list, axis_length=1):
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x", size=15, color="black")
    ax.set_ylabel("z", size=15, color="black")
    ax.set_zlabel("y", size=15, color="black")

    color_list = ["r", "g", "b"]
    for M in M_list:
        o = M[0:3, 3]
        for i in range(3):
            p = o + axis_length * M[0:3, i]
            P = np.array([o, p]).T
            ax.plot(P[0], P[2], P[1], color=color_list[i])

    ax.set_aspect('equal')
    ax.invert_zaxis()
    plt.show()


def plot_points(P):
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x", size=15, color="black")
    ax.set_ylabel("z", size=15, color="black")
    ax.set_zlabel("y", size=15, color="black")

    ax.scatter(P[0], P[2], P[1], s=1, color="r")

    ax.set_aspect('equal')
    ax.invert_zaxis()
    plt.show()


def sfm_by_orthogonal_three_points(c1_P_dir, c1_t_c2):
    print(c1_P_dir)
    print(c1_t_c2)
    s = np.linalg.norm(c1_t_c2[:, 0])
    c1_t_c2 = c1_t_c2 / s
    p1xt = np.cross(c1_P_dir[:, 0], c1_t_c2[:, 0])
    p2xt = np.cross(c1_P_dir[:, 1], c1_t_c2[:, 0])
    p3xt = np.cross(c1_P_dir[:, 2], c1_t_c2[:, 0])
    print(np.linalg.norm(np.cross(p1xt, p2xt)), np.linalg.norm(np.cross(p2xt, p3xt)), np.linalg.norm(np.cross(p3xt, p1xt)))
    if (np.linalg.norm(np.cross(p1xt, p2xt)) == 0 or np.linalg.norm(np.cross(p2xt, p3xt)) == 0 or np.linalg.norm(np.cross(p3xt, p1xt)) == 0):
        raise Exception("invalid input pi x t // pj x t")

    # C1からの方向ベクトルと並進ベクトルからスケールを決定する
    a = np.dot(c1_P_dir[:, 0], c1_P_dir[:, 1])
    b = np.dot(c1_P_dir[:, 1], c1_P_dir[:, 2])
    c = np.dot(c1_P_dir[:, 2], c1_P_dir[:, 0])
    d = -np.dot(c1_P_dir[:, 0], c1_t_c2[:, 0])
    e = -np.dot(c1_P_dir[:, 1], c1_t_c2[:, 0])
    f = -np.dot(c1_P_dir[:, 2], c1_t_c2[:, 0])
    g = np.dot(c1_t_c2[:, 0], c1_t_c2[:, 0])

    Q_candiate = []

    print("a,b,c,d,e,f,g", a, b, c, d, e, f, g)

    if len(Q_candiate) == 0:
        # q1についての2次方程式を解く(cx+f)(Ax^2+Bx+C) = 0
        A = b*d*d - c*e*d + c*a*g - f*a*d
        B = 2*(b*d*g - e*f*d)
        C = b*g*g - e*f*g

        print("cfABC", c, f, A, B, C)
        print("B * B - 4 * A * C", B * B, - 4 * A * C)
        if (c != 0 or f != 0) and (A != 0 or B != 0 or C != 0):  # (cx+f)(Ax^2+Bx+C) = 0 が自明でない場合
            x = []
            if A != 0 and B * B - 4 * A * C >= 0:
                x.append((-B + np.sqrt(B * B - 4 * A * C)) / (2 * A))
                x.append((-B - np.sqrt(B * B - 4 * A * C)) / (2 * A))
            elif B != 0:
                x.append(-C/B)
            if c != 0:
                x.append(-f/c)
            print("q1", x)
            x = list(set(x))  # 重複を削除
            for q1 in x:
                print("a, q1, e, c, q1, f:", a, q1, e, c, q1, f)
                print(a*q1 + e, c*q1 + f)
                if (a*q1 + e != 0 and c*q1 + f != 0):
                    q2 = -(d*q1 + g)/(a*q1 + e)
                    q3 = -(d*q1 + g)/(c*q1 + f)
                    Q_candiate.append(np.array([q1, q2, q3]))
                elif (a*q1 + e != 0):
                    q2 = -(d*q1 + g)/(a*q1 + e)
                    if (b*q2 + f != 0):
                        q3 = -(e*q2 + g)/(b*q2 + f)
                        Q_candiate.append(np.array([q1, q2, q3]))
                elif (c*q1 + f != 0):
                    q3 = -(d*q1 + g)/(c*q1 + f)
                    if (b*q3 + e != 0):
                        q2 = -(f*q3 + g)/(b*q3 + e)
                        Q_candiate.append(np.array([q1, q2, q3]))

    if len(Q_candiate) == 0:
        # q2についての2次方程式を解く(ax+d)(Ax^2+Bx+C) = 0
        A = c*e*e - a*f*e + a*b*g - d*b*e
        B = 2*(c*e*g - f*d*e)
        C = c*g*g - f*d*g
        if (a != 0 or d != 0) and (A != 0 or B != 0 or C != 0):  # (ax+d)(Ax^2+Bx+C) = 0 が自明でない場合
            x = []
            if A != 0 and B * B - 4 * A * C >= 0:
                x.append((-B + np.sqrt(B * B - 4 * A * C)) / (2 * A))
                x.append((-B - np.sqrt(B * B - 4 * A * C)) / (2 * A))
            elif B != 0:
                x.append(-C/B)
            if a != 0:
                x.append(-d/a)
            x = list(set(x))  # 重複を削除
            print("q2", x)
            for q2 in x:
                if (b*q2 + f != 0 and a*q2 + d != 0):
                    q3 = -(e*q2 + g)/(b*q2 + f)
                    q1 = -(e*q2 + g)/(a*q2 + d)
                    Q_candiate.append(np.array([q1, q2, q3]))
                elif (b*q2 + f != 0):
                    q3 = -(e*q2 + g)/(b*q2 + f)
                    if (c*q3 + d != 0):
                        q1 = -(f*q3 + g)/(c*q3 + d)
                        Q_candiate.append(np.array([q1, q2, q3]))
                elif (a*q2 + d != 0):
                    q1 = -(e*q2 + g)/(a*q2 + d)
                    if (c*q1 + f != 0):
                        q3 = -(d*q1 + g)/(c*q1 + f)
                        Q_candiate.append(np.array([q1, q2, q3]))

    if len(Q_candiate) == 0:
        # q3についての2次方程式を解く(bx+e)(Ax^2+Bx+C) = 0
        A = a*f*f - b*d*f + b*c*g - e*c*f
        B = 2*(a*f*g - d*e*f)
        C = a*g*g - d*e*g
        if (b != 0 or e != 0) and (A != 0 or B != 0 or C != 0):  # (bx+e)(Ax^2+Bx+C) = 0 が自明でない場合
            x = []
            if A != 0 and B * B - 4 * A * C >= 0:
                x.append((-B + np.sqrt(B * B - 4 * A * C)) / (2 * A))
                x.append((-B - np.sqrt(B * B - 4 * A * C)) / (2 * A))
            elif B != 0:
                x.append(-C/B)
            if b != 0:
                x.append(-e/b)
            x = list(set(x))  # 重複を削除
            print("q3", x)
            for q3 in x:
                if (c*q3 + d != 0 and b*q3 + e != 0):
                    q1 = -(f*q3 + g)/(c*q3 + d)
                    q2 = -(f*q3 + g)/(b*q3 + e)
                    Q_candiate.append(np.array([q1, q2, q3]))
                elif (c*q3 + d != 0):
                    q1 = -(f*q3 + g)/(c*q3 + d)
                    if (a*q1 + e != 0):
                        q2 = -(d*q1 + g)/(a*q1 + e)
                        Q_candiate.append(np.array([q1, q2, q3]))
                elif (b*q3 + e != 0):
                    q2 = -(f*q3 + g)/(b*q3 + e)
                    if (a*q1 + e != 0):
                        q1 = -(e*q2 + g)/(a*q2 + d)
                        Q_candiate.append(np.array([q1, q2, q3]))

    print("Q", Q_candiate)
    score = []
    result_candidate = []
    for Q in Q_candiate:
        c1_P = Q * c1_P_dir
        print(c1_P)
        c1_P_T = c1_P - c1_t_c2
        # 十分性チェック
        cos12 = np.abs(np.dot(c1_P_T[:, 0], c1_P_T[:, 1]) / (np.linalg.norm(c1_P_T[:, 0]) * np.linalg.norm(c1_P_T[:, 1])))
        cos23 = np.abs(np.dot(c1_P_T[:, 1], c1_P_T[:, 2]) / (np.linalg.norm(c1_P_T[:, 1]) * np.linalg.norm(c1_P_T[:, 2])))
        cos31 = np.abs(np.dot(c1_P_T[:, 2], c1_P_T[:, 0]) / (np.linalg.norm(c1_P_T[:, 2]) * np.linalg.norm(c1_P_T[:, 0])))
        score.append(cos12 + cos23 + cos31)

    min_index = int(np.array(score).argmin())
    c1_P = s * Q_candiate[min_index] * c1_P_dir
    c1_P_T = c1_P - s * c1_t_c2
    c1_R_c2 = c1_P_T / np.linalg.norm(c1_P_T, axis=0)
    c1_M_c2 = pose_Rt2M(c1_R_c2, s * c1_t_c2)
    c2_P = homogeneous_transform(np.linalg.inv(c1_M_c2), c1_P)
    return [c1_P, c2_P, c1_M_c2]


# R = Rotation.from_rotvec([0, 0, 0]).as_matrix()
# t = np.array([[30, 0, 10]]).T
# # R = np.eye(3)
# # t = np.array([[0, 0, 1]]).T
# T = np.vstack([np.hstack([R, t]), np.array([[0, 0, 0, 1]])])

# p1 = np.array([[-1.0525725,   0.,          0.],
#                [0.,         0.51283875, - 0.51283875],
#                [1.,        1.,          1.]])
# t = np.array([[0.],
#               [0.],
#               [500]])

# p1 = np.array([[-1, 0, 0],
#                [0., 1, -1],
#                [1., 2, 2]])
# t = np.array([[0.],
#               [0.],
#               [500]])


# # print(T)
# # p2 = np.array([[10, 0, 0], [0, 500, 0], [0, 0, 100]]).T
# # p1 = utils.homogeneous_transform(T, p2)


# [c1_p_est, c2_p_est, R_est] = sfm_by_orthogonal_three_points(p1, t)
# # [c1_p_est, c2_p_est, R_est] = utils.sfm_by_orthogonal_three_points(p1, t)

# print("c1_p_est", c1_p_est)
# print("c2_p_est", c2_p_est)
# print("R_est", R_est)
