import numpy as np

def strided_app(a, L, S):  
    """
    Function to split audio as overlapping segments
    L : Window length
    S : Stride (L/stepsize)
    """
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a,\
        shape=(nrows,L),\
        strides=(S*n,n))