import numpy as np
class AnomalyDetectionAlgorithm():
    def __init__(self,epsilon=None):
        self.__epsilon=epsilon
        self.__mu=None
        self.__var=None
        self.__f1_score=None
    def fit(self,x):
        self.mu = np.zeros(x.shape[1])
        self.var = np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            self.mu[i]=np.mean(x[:,i])
            self.var[i]=np.mean((x[:,i]-self.mu[i])**2)   
    def get_probability (self,x):
        if (self.mu is None or self.var is None):
            raise ValueError('Model parameters mu and var must be initialized by calling fit.')
        p_tot=np.ones(x.shape[0])
        for j in range(x.shape[1]):
             if self.var[j] == 0: 
                p=np.ones(x.shape[0])
             else:
                p=(1/np.sqrt(2*np.pi*self.var[j])) * ( np.exp(-1 * ((x[:,j]-self.mu[j])**2 / (2*self.var[j]) )) )
             p_tot*=p
        return p_tot
    def predict(self,x_t):
        if self.epsilon is None:
            raise ValueError('Epsilon must be assignedp_tot before making predictions.')
        if self.mu is None or self.var is None:
            raise ValueError('Model parameters mu and var must be initialized by calling fit.')
        p_tot = self.get_probability(x_t)
        p_fin = (p_tot < self.epsilon).astype(int)
        return p_fin
    def select_epsilon (self,x_val,y_val):
        if self.mu is None or self.var is None:
            raise ValueError('The model needs to be fitted with training data before selecting epsilon.')
        best_epsilon = 0
        best_F1 = 0
        F1 = 0
        p_val = self.get_probability(x_val)
        step_size = (max(p_val) - min(p_val)) / 1000
        for epsilon in np.arange(min(p_val), max(p_val), step_size):
            p_fin = (p_val < epsilon).astype(int)
            tp = np.sum((p_fin == 1) & (y_val == 1))
            fp = np.sum((p_fin == 1) & (y_val == 0))
            fn = np.sum((p_fin == 0) & (y_val == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            F1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            if F1 > best_F1:
                best_F1 = F1
                best_epsilon = epsilon
        self.epsilon = best_epsilon
        self.f1_score = best_F1
        return self.epsilon,self.f1_score 
