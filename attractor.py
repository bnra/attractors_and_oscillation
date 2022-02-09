import numpy as np


def sim_vec(matrix: np.ndarray):
    """
    similarity of row vectors

    r_ij = dot product of (row) vectors i,j of length l divided by l,
    where matrix is of dimensions (n,l)
    """
    return matrix @ matrix.T / matrix.shape[1]




def compute_conductance_scaling(patterns: np.ndarray, sparsity: float):
    """
    compute the scaling factor of the conductance  according to Battaglia, Treves 1998
    (https://pubmed.ncbi.nlm.nih.gov/9472489/)
    original process: i) g_ij := 0 ii) for each pattern do:
        a) delta_g_ij = g_EE / C_EE * (n_i^p / sparsity - 1) (n_j^p / sparsity - 1) b) g_ij = max(0, g_ij + delta_g_ij)
    where g_ij is the conductance of synapse from neuron with index i to j

    Here scaling factor s is computed:
    - g = g_EE / C_EE * s
    - process: i) s_ij := 0 ii) for each pattern do:
        a) delta_s_ij = (n_i^p / sparsity - 1) (n_j^p / sparsity - 1) b) s_ij = max(0, s_ij + delta_s_ij)
    (clipping equivalent to original process as g_EE/C_EE is a positive constant term therefore
        crossing of 0 (clipping) remains unchanged)


    :param patterns: patterns tb used in compution shape: (p,size) where size is the size of the pattern
                    (= size of E population) and p is the number of patterns
    :param sparsity: sparsity of the patterns
    :return: scaling factor s for conductances (shape: (size,size))
    """

    if sparsity < 0.0 or sparsity > 1.0:
        raise ValueError(f"sparsity must be in [0,1]. Is {sparsity}.")

    size = patterns.shape[1]
    s = np.zeros(size * size).reshape(size, size)

    for p in range(patterns.shape[0]):
        pattern = patterns[p] / sparsity - 1
        delta_s = np.outer(pattern, pattern)
        s = np.maximum(0, delta_s + s)

    return s


def compute_conductance_scaling_single_clip(patterns: np.ndarray, sparsity: float):
    """
    compute the scaling factor of the conductance by summing over patterns and clipping the result

    Here scaling factor s is computed:
    - g = g_EE / C_EE * s
    - process: i) for each pattern compute s^p_ij = (n_i^p / sparsity - 1) (n_j^p / sparsity - 1)
              ii) s_ij = sum_p s^p_ij
              iii) s_ij = max(0, s_ij)


    :param patterns: patterns tb used in compution shape: (p,size) where size is the size of the pattern
                    (= size of E population) and p is the number of patterns
    :param sparsity: sparsity of the patterns
    :return: scaling factor s for conductances (shape: (size,size))
    """

    if sparsity < 0.0 or sparsity > 1.0:
        raise ValueError(f"sparsity must be in [0,1]. Is {sparsity}.")

    # patterns: shape (num_p, size) one row ~ a pattern; ij,ib->jb: i) create outer product for each row ii) sum over rows
    # (np.sum(np.vstack([np.outer(p,p).reshape(1,p.size, p.size) for p in patterns / sparsity - 1]), axis=0))
    pat = patterns / sparsity - 1
    s = np.einsum("ij,ib->jb", pat, pat)
    return np.maximum(0, s)


def compute_conductance_scaling_unclipped(patterns: np.ndarray, sparsity: float):
    """
    compute the scaling factor of the conductance by summing over patterns

    Here scaling factor s is computed:
    - g = g_EE / C_EE * s
    - process: i) for each pattern compute s^p_ij = (n_i^p / sparsity - 1) (n_j^p / sparsity - 1)
              ii) s_ij = sum_p s^p_ij


    :param patterns: patterns tb used in compution shape: (p,size) where size is the size of the pattern
                    (= size of E population) and p is the number of patterns
    :param sparsity: sparsity of the patterns
    :return: scaling factor s for conductances (shape: (size,size))
    """

    if sparsity < 0.0 or sparsity > 1.0:
        raise ValueError(f"sparsity must be in [0,1]. Is {sparsity}.")

    pat = patterns / sparsity - 1
    return np.einsum("ij,ib->jb", pat, pat)


def normalize(matrix: np.ndarray, frm: float = None, to: float = None):
    """
    normalize (here squash) all values in matrix to [0,1] and if specified rescale to [frm,to]

    :param matrix: matrix tb normalized
    :param frm: lower bound of interval to which matrix is tb rescaled (requires setting to)
    :param to: upper bound of interval to which matrix is tb rescaled (requires setting frm)
    :return: matrix normalized to [0,1] or [frm,to] if specified
    """
    if (frm == None or to == None) and frm != to:
        raise ValueError("frm and to must either both be specified or neither.")
    mn = np.min(matrix)
    mx = np.max(matrix)
    norm_m = (matrix - mn) / (mx - mn)
    return norm_m * (to - frm) + frm if frm != None and to != None else norm_m


def z_score(matrix: np.ndarray):
    """
    compute z score: z := (x - mu) / sigma
    where mu = mean(matrix), sigma = std(matrix) (over all values in matrix),
    for all values x in matrix

    :param matrix: matrix tb normalized
    :return: z_score of the matrix
    """
    return (matrix - np.mean(matrix)) / np.std(matrix)
