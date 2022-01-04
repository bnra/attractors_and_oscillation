import numpy as np

def compute_conductance_scaling(patterns:np.ndarray, sparsity:float):
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
    s = np.zeros(size * size). reshape(size, size)

    for p in range(patterns.shape[0]):
        pattern = patterns[p] / sparsity - 1
        delta_s = np.outer(pattern, pattern)
        s = np.maximum(0, delta_s + s)
    
    return s
