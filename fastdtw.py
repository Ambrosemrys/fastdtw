import time
import warnings
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from numpy import linalg
from scipy import sparse


def check_euclidean_inputs(X, Y):
    """
    Check the input of two time series in Euclidean spaces, which
    are to be warped to each other.  They must satisfy:
    1. They are in the same dimension space
    2. They are 32-bit
    3. They are in C-contiguous order

    If #2 or #3 are not satisfied, automatically fix them and
    warn the user.
    Furthermore, warn the user if X or Y has more columns than rows,
    since the convention is that points are along rows and dimensions
    are along columns

    Parameters
    ----------
    X: ndarray(M, d)
        The first time series
    Y: ndarray(N, d)
        The second time series

    Returns
    -------
    X: ndarray(M, d)
        The first time series, possibly copied in memory to be 32-bit, C-contiguous
    Y: ndarray(N, d)
        The second time series, possibly copied in memory to be 32-bit, C-contiguous
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("The input time series are not in the same dimension space")
    if not X.dtype == np.float32:
        warnings.warn("X is not 32-bit, so creating 32-bit version")
        X = np.array(X, dtype=np.float32)
    if not Y.dtype == np.float32:
        warnings.warn("Y is not 32-bit, so creating 32-bit version")
        Y = np.array(Y, dtype=np.float32)
    return X, Y


def fill_block(A, p, radius, val):
    """
    Fill a square block with values

    Parameters
    ----------
    A: ndarray(M, N) or sparse(M, N)
        The array to fill
    p: list of [i, j]
        The coordinates of the center of the box
    radius: int
        Half the width of the box
    val: float
        Value to fill in
    """

    move_path = []
    move_path =sum([[2*x,x+y] for x, y in zip(p[:-1],p[1:])],[])
    move_path.append(2*p[-1])
    # projecting
    for path_start in move_path:
        A[path_start[0]:path_start[0] + 2, path_start[1]:path_start[1] + 2] = val
    # expanding radius
    in_radius_index = []
    for r in range(1, radius + 1):
        bottom_left = np.clip(np.transpose(A.nonzero()) + [r, -r],0,A.shape[0]-1)
        in_radius_index.extend(bottom_left.tolist())
        top_right = np.clip(np.transpose(A.nonzero()) + [-r, r],0, A.shape[1] - 1)
        in_radius_index.extend(top_right.tolist())
    for index in in_radius_index:
        A[index[0], index[1]] = max(0.5, A[index[0], index[1]])

def calculate_distance(series_a, series_b, method='l-2'):
    # use lp distance, with 1
    p = method.split("-")[-1]
    if p.isdigit():
        distance = linalg.norm(series_a - series_b, ord=int(p))
    elif p == 'inf':
        distance = (linalg.norm(series_a - series_b, ord=np.inf))
    return distance


def fastdtw_dynstep(X, Y,Occ=None, distance_method='l-2'):
    """
    input
        X: ndarray(M, d)
            A d-dimensional Euclidean point cloud with M points
        Y: ndarray(N, d)
            A d-dimensional Euclidean point cloud with N points
    return: {'cost': S[-1], 'P': P}
    """
    M = X.shape[0]
    N = Y.shape[0]

    S = sparse.lil_matrix((M, N))
    P = sparse.lil_matrix((M, N), dtype=int)
    if Occ == None:
        Occ = np.ones((M,N))
    I, J = Occ.nonzero()
    ## Step 2: Find indices of left, up, and diag neighbors.
    idx = sparse.coo_matrix((np.arange(I.size) + 1, (I, J)), shape=(M + 1, N + 1)).tocsr()
    # idx is all the indices of all non-zero elements in the matrix.
    # Left neighbors
    left = np.array(idx[I, J - 1], dtype=np.int32).flatten()
    left[left <= 0] = -1
    left -= 1
    # Up neighbors
    up = np.array(idx[I - 1, J], dtype=np.int32).flatten()
    up[up <= 0] = -1
    up -= 1
    # Diag neighbors
    diag = np.array(idx[I - 1, J - 1], dtype=np.int32).flatten()
    diag[diag <= 0] = -1
    diag -= 1
    ## Step 3: Pass information for dynamic programming steps
    S = np.zeros(I.size, dtype=np.float32)  # Dyn prog matrix
    P = np.zeros(I.size, dtype=np.int32)  # Path pointer matrix
    cnt = left.size
    for idx in range(cnt):
        # Step 1: Compute Euclidean distance
        # calculate_distance is a self defined distance metric, that take two data point or two vector with equal length and return a scaler.
        dist = calculate_distance(X[I[idx]], Y[J[idx]], distance_method)
        # Step 2: Do dynamic programming step
        score = -1  # initialize to -1
        LEFT = 0
        UP = 1
        DIAG = 2
        if idx == 0:
            score = 0
        else:
            left_score = -1
            if left[idx] >= 0:
                left_score = S[left[idx]]
            up_score = -1
            if up[idx] >= 0:
                up_score = S[up[idx]]
            diag_score = -1
            if diag[idx] >= 0:
                diag_score = S[diag[idx]]
            if left_score > -1:
                score = left_score
                P[idx] = LEFT
            if (up_score > -1) and (up_score <= score or score == -1):
                score = up_score
                P[idx] = UP
            if (diag_score > -1) and (diag_score <= score or score == -1):
                score = diag_score
                P[idx] = DIAG
        S[idx] = score + dist
    P = sparse.coo_matrix((P, (I, J)), shape=(M, N)).tocsr()
    i = M-1
    j = N-1
    path = [[i, j]]
    step = [[0, -1], [-1, 0], [-1, -1]]  # LEFT, UP, DIAG
    while not (path[-1][0] == 0 and path[-1][1] == 0):
        s = step[P[i, j]]
        i += s[0]
        j += s[1]
        path.append([i, j])
    path.reverse()
    path = np.array(path, dtype=int)
    ret = {'cost': S[-1], 'P': path}
    return ret



def dtw_brute_backtrace(X, Y, Occ = None, level=0, do_plot=False):
    """
    DTW on a constrained occupancy mask.  A helper method for  fastdtw


    Parameters
    ----------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    Occ: scipy.sparse((M, N))
        A MxN array with 1s if this cell is to be evaluated and 0s otherwise

    level: int
        An int for keeping track of the level of recursion, if applicable
    do_plot: boolean
        Whether to plot the warping path at each level and save to image files

    Returns
    -------
        (float: cost, ndarray(K, 2): The warping path)
    """
    X, Y = check_euclidean_inputs(X, Y)
    ret = fastdtw_dynstep(X, Y, Occ)
    if do_plot:
        plot_dtw(X.shape[0],Y.shape[0],ret['P'],level,Occ)
    return (ret['cost'], ret['P'])


def plot_dtw(M,N,path,level,Occ=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = colors.ListedColormap(['white', '#454545'])
    if Occ == None:
        Occ = np.ones((M, N))
        usage = "original dtw"
    else:
        Occ = Occ.toarray()
        usage = "fast dtw"
    ax.imshow(Occ, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1.5)
    ax.set_xticks(np.arange(-.5, Occ.shape[0], 1))
    ax.set_yticks(np.arange(-.5, Occ.shape[1], 1))

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    path = np.array(path)
    plt.plot(path[:, 1], path[:, 0], c='black', linewidth=2, label="warp path")
    line_legend = mlines.Line2D([], [], color="black", label="warp path")

    if usage == "original dtw":
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for i in range(Occ.shape[0]):
            ax.text(-1, i, str(i), ha='center', va='center', color='black')
        for j in range(Occ.shape[1]):
            ax.text(j, Occ.shape[0], str(j), ha='center', va='center', color='black')
        black_patch = mpatches.Patch(facecolor="#454545", label="search area", edgecolor="black")
        ax.legend(handles=[black_patch, line_legend], bbox_to_anchor=(1.05, 1),
                  loc="upper left")
        plt.title("Dynamic Warping Paths")
        plt.savefig("DTW plot.png", bbox_inches='tight')
    else:
        grey_patch = mpatches.Patch(facecolor="grey", label="search window-by radius", edgecolor="black")
        black_patch = mpatches.Patch(facecolor="#454545", label="search window-by projection", edgecolor="black")
        white_patch = mpatches.Patch(facecolor="white", label="unsearched area", edgecolor="black")
        ax.legend(handles=[grey_patch, black_patch, white_patch, line_legend], bbox_to_anchor=(1.05, 1),
                  loc="upper left")
        plt.title("Last Level")
        plt.savefig("Last Level.pdf", bbox_inches='tight')
def reduce_by_half(series):
    """
    Reduce a time series by half by taking the average value of every consecutive pairs.
    series: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    return: A d-dimensional Euclidean point cloud with M//2 points

    """
    X = series.copy()
    if X.shape[0] % 2 != 0:
        X[-2] = (X[-2] + X[-1]) / 2
        X = X[:-1]
    return (X[1::2, ] + X[::2, ]) / 2


def fastdtw(X, Y, radius, level=0, do_plot=False):
    """
    An implementation of [1]
    [1] FastDTW: Toward Accurate Dynamic Time Warping in Linear Time and Space. Stan Salvador and Philip Chan

    Parameters
    ----------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    radius: int
        Radius of the l-infinity box that determines sparsity structure
        at each level
    level: int
        An int for keeping track of the level of recursion
    do_plot: boolean
        Whether to plot the warping path at each level and save to image files

    Returns
    -------
        (float: cost, ndarray(K, 2): The warping path)
    """
    X, Y = check_euclidean_inputs(X, Y)
    minTSsize = radius + 2
    M = X.shape[0]
    N = Y.shape[0]
    X = np.ascontiguousarray(X)
    Y = np.ascontiguousarray(Y)
    if M < minTSsize or N < minTSsize:
        return dtw_brute_backtrace(X, Y, do_plot=do_plot)
    # Recursive step
    shrunk_x = reduce_by_half(X)
    shrunk_y = reduce_by_half(Y)
    cost, path = fastdtw(shrunk_x, shrunk_y, radius, level + 1, do_plot)
    # cost, path = fastdtw(X[0::2, :], Y[0::2, :], radius,  level + 1, do_plot)
    if type(path) is dict:
        path = path['path']
    path = np.array(path)
    Occ = sparse.lil_matrix((M, N))

    fill_block(Occ, path, radius, 1)
    return dtw_brute_backtrace(X, Y, Occ, level, do_plot)

def dtw_similarity(X,Y):
    cost_dtw, path_dtw = dtw_brute_backtrace(X, Y, do_plot=True)
    similarity = 1 - cost_dtw/calculate_distance(X,Y)
    return similarity 

if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(16, 1)
    Y = np.random.rand(16, 1)
    tic = time.time()
    cost_fast, path_fast = fastdtw(X, Y, radius=2, level=0, do_plot=True)
    toc = time.time()
    print(
        "Implementing FastDTW, on X with {} data points and Y with {} data points.\n-Time Use: {}s\n-Distance: {}".format(
            X.shape[0], Y.shape[0], toc - tic, cost_fast))
    tic = time.time()
    cost_dtw, path_dtw = dtw_brute_backtrace(X, Y, do_plot=True)
    toc = time.time()
    print(
        "Implementing OriginalDTW, on X with {} data points and Y with {} data points.\n-Time Use: {}s\n-Distance: {}".format(
            X.shape[0], Y.shape[0], toc - tic, cost_dtw))

