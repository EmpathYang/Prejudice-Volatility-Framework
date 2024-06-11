import numpy as np

def d_y(y: int, class_num: int = 2):
    if y + 1 > class_num:
        raise ValueError("Index y should be smaller than the class number.")
    _d_y = np.ones((class_num)) * -1 / (class_num - 1)
    _d_y[y] = 1
    return _d_y

def S_y(d_y: np.ndarray, P: np.ndarray):
    return np.tensordot(P, d_y, axes=([-1], [0]))

def J(P: np.ndarray, norm: int = np.inf):
    if (P < 0).any():
        raise ValueError("P should be of positive elements.")
    class_num = P.shape[-1]
    ds = [d_y(y=i, class_num=class_num) for i in range(class_num)]
    R = np.stack([S_y(d_y=d, P=P) for d in ds], axis=-1)
    
    R = np.linalg.norm(R, axis=-1, ord=norm) / np.linalg.norm(ds[0], ord=norm)
    return R
    
def compute_r(P_matrix: np.ndarray, T_distribution: np.ndarray):
    R_matrix = J(P_matrix)
    return np.matmul(R_matrix, T_distribution)

def compute_r_prejudice(P_matrix: np.ndarray, T_distribution: np.ndarray):
    _T_distribution = np.expand_dims(np.expand_dims(T_distribution, axis=0), axis=-1)
    P_vector = np.sum(P_matrix*_T_distribution, axis=-2)
    return J(P_vector)

def compute_r_volatility(P_matrix: np.ndarray, T_distribution: np.ndarray):
    _r = compute_r(P_matrix, T_distribution)
    _r_prejudice = compute_r_prejudice(P_matrix, T_distribution)
    return _r - _r_prejudice

def compute_R(r_vector: np.ndarray, X_distribution: np.ndarray):
    return np.dot(r_vector, X_distribution)

def compute_risk(P_matrix: np.ndarray, T_distribution: np.ndarray, X_distribution: np.ndarray):
    r_X = compute_r(P_matrix, T_distribution)
    r_X_prejudice = compute_r_prejudice(P_matrix, T_distribution)
    r_X_volatility = compute_r_volatility(P_matrix, T_distribution)
    R = compute_R(r_X, X_distribution)
    R_prejudice = compute_R(r_X_prejudice, X_distribution)
    R_volatility = compute_R(r_X_volatility, X_distribution)
    return r_X, r_X_prejudice, r_X_volatility, R, R_prejudice, R_volatility

if __name__ == "__main__":
    P_matrix = np.array(
        [[[0.58, 0.42],
        [0.47, 0.53],
        [0.34, 0.66],
        [0.57, 0.43]],

        [[0.47, 0.53],
        [0.47, 0.53],
        [0.93, 0.07],
        [0.9,  0.1 ]],

        [[0.7,  0.3 ],
        [0.19, 0.81],
        [0.58, 0.42],
        [0.44, 0.56]],

        [[0.33, 0.67],
        [0.61, 0.39],
        [0.43, 0.57],
        [0.44, 0.56]]])
    T_distribution = np.array([0.4, 0.3, 0.2, 0.1])
    X_distribution = np.array([0.45, 0.35, 0.15, 0.05])
    r_X, r_X_prejudice, r_X_volatility, R, R_prejudice, R_volatility = compute_risk(P_matrix=P_matrix, T_distribution=T_distribution, X_distribution=X_distribution)
    print(r_X, r_X_prejudice, r_X_volatility, R, R_prejudice, R_volatility)
