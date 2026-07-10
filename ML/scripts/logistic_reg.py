import numpy as np
import scipy

def logistic_regression_loss(theta, D, L_one_hot, lmbda):
    # 1. Spacchetta theta in W (5x6) e b (6x1)
    W=theta[0:30].reshape(5,6)
    b=theta[30:36].reshape(6,1)
    N=D.shape[1]
    # 2. Calcola gli score S = W.T @ D + b
    S=W.T @ D + b
    # 3. Applica la Softmax stabile per ottenere la matrice delle probabilità P
    s_stab=S-np.max(S,axis=0)
    softmax=np.exp(s_stab)/np.sum(np.exp(s_stab),axis=0)
    # 4. Calcola il valore della Loss J (ricorda la regolarizzazione L2 solo su W, b di solito non si regolarizza!)
    J=lmbda*0.5 * np.sum(W**2) - (1/N) * np.sum(L_one_hot*np.log(softmax))
    # 5. Calcola i gradienti grad_W e grad_b
    grad_W= lmbda*W + (1/N)*D @ (softmax - L_one_hot).T
    grad_b= (1/N)* np.sum(softmax-L_one_hot,axis=1)
    grad_flat= np.hstack([grad_W.ravel(),grad_b.ravel()])
    # 6. Ritorna (J, grad_flat)
    return J,grad_flat
def get_L_one_hot(D,L):
    L_one_hot=[]
    for c in np.unique(L):
        zeros=np.where(L==c,1,0)
        L_one_hot.append(zeros)
    return np.array(L_one_hot)
def train_log_reg(D,L,lmbda):

    theta_init=np.zeros(36)
    L_one_hot=get_L_one_hot(D,L)
    # Delega dell'addestramento all'ottimizzatore L-BFGS-B
    theta_opt, f_opt, dict_opt = scipy.optimize.fmin_l_bfgs_b(
        func=logistic_regression_loss,
        x0=theta_init,
        args=(D, L_one_hot, lmbda)
    )
    
    # Estrazione e ridimensionamento dei parametri ottimali per l'inferenza
    W_opt = theta_opt[0:30].reshape(5, 6)
    b_opt = theta_opt[30:36].reshape(6, 1)
    
    return W_opt, b_opt
if __name__ == "__main__":
    print("Logistic Regression Scripts")