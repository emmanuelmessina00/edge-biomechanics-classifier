import numpy as np
from scripts.utility import getMu, vCol,vRow, compute_Sb_Sw
import scipy

def LDA(X,L,m):
    Sb,Sw=compute_Sb_Sw(X,L)
    _,U=scipy.linalg.eigh(Sb,Sw)
    W=U[:,::-1][:,:m]
    return W.T @ X

if __name__ == "__main__":

    print("LDA Script")