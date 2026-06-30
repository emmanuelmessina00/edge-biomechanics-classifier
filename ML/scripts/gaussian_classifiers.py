import numpy as np
from scripts.utility import getC,getMu,vCol,compute_Sb_Sw

#Functions for Gaussian Multivariate
def train_Gaussian_multivariate(D,L):
    parameters=[]

    for l in np.unique(L):
        mask=(L==l)
        D0=D[:,mask]
        mu=getMu(D0)
        C=getC(D0)
        parameters.append((mu,C))
    return parameters

def get_priors(D,L):
    N=D.shape[1]
    priors=[]
    for l in np.unique(L):
        mask=(L==l)
        D0=D[:,mask]
        NC=D0.shape[1]
        priors.append(NC/N)
    
    return priors
def get_logdensity_GM_dist_by_params(D,mu,C):

    Dim=D.shape[0]
    log_C=np.linalg.slogdet(C)[1]
    inv_C=np.linalg.inv(C)
    DC=D- vCol(mu)

    return - Dim * 0.5 * np.log(2*np.pi) - 0.5*log_C-0.5*np.sum((inv_C @ DC *DC),axis=0)

def get_logdensity_GM(D,L,parameters):

    logdensities=[]

    for idx,l in enumerate(np.unique(L)):
        mu0,C0=parameters[idx]
        logdens=get_logdensity_GM_dist_by_params(D,mu0,C0)
        logdensities.append(logdens)
    
    S=np.vstack(logdensities)
    return S

def getSpost(D,L,parameters,priors):
    S=get_logdensity_GM(D,L,parameters)
    return S + vCol(np.log(priors))

def get_predictions_Gaussian_Mult(D,L,parameters_GM,priors):
    SPOST=getSpost(D,L,parameters_GM,priors) 
    predictions= np.argmax(SPOST,axis=0)
    return predictions

def train_Gaussian_Naive_Bayes(D,L):
    parameters=[]

    for l in np.unique(L):
        mask=(L==l)
        D0=D[:,mask]
        mu=getMu(D0)
        C=getC(D0)
        Diagonal_C=np.eye(C.shape[1])*C
        parameters.append((mu,Diagonal_C))
    return parameters

def train_Gaussian_TiedCovariance(D,L):
    parameters=[]
    _,cov=compute_Sb_Sw(D,L)
    for l in np.unique(L):
        mask=(L==l)
        D0=D[:,mask]
        mu=getMu(D0)
        parameters.append((mu,cov))
    return parameters
if __name__ == "__name__":
    print("Gaussian Classifiers Scripts")