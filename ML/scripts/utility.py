import numpy as np

def vRow(vet):
    return vet.reshape((1, vet.size))
def vCol(vet):
     return vet.reshape((vet.size, 1))
def getMu(D):
    return D.sum(axis=1)/D.shape[1]

def compute_Sb_Sw(D, L):
    Sw=0
    Sb=0
    N=D.shape[1]
    mu=vCol(getMu(D))
    for c in np.unique(L):
        mask=L==c
        D0=D[:,mask]
        nc=D0.shape[1]
        mc=vCol(getMu(D0))
        Sb+=nc*((mc-mu)@(mc-mu).T)
        DC=D0-mc
        Sw+=(DC @ DC.T)
    return Sb/N,Sw/N

def map_labels_into_numbers(labels):

    unique_labels, labels_n = np.unique(labels, return_inverse=True)
    
    labels_to_num = {l: idx for idx, l in enumerate(unique_labels)}
    num_to_label = {idx: l for idx, l in enumerate(unique_labels)}
    
    return labels_to_num,num_to_label,labels_n

if __name__ == "__main__":

    print("Utility Functions")