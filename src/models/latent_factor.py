##################################################
# latent factor model                                #
##################################################
from typing import Any, Dict, List, Tuple
from collections import defaultdict
import numpy as np


class latent_factor_model:
    # still only requires u,i,r as input

    def __init__(
        self,
        lr ,
        n_epochs,
        shuffle,
        seed,
        reg,
        n_factors,
    ):

        self.lr = lr
        self.n_epochs = n_epochs
        self.reg = reg
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.n_factors = int(n_factors)

        self.alpha = 0.0
        self.beta_U = defaultdict(float)
        self.beta_I = defaultdict(float)

        self.P = {}
        self.Q = {}
        
        


    def _init_user(self,u):
        if u not in self.P:
            self.P[u] = self.rng.normal(0.0,0.1,self.n_factors)
    
    def _init_item(self,i):
        if i not in self.Q :
            self.Q[i] = self.rng.normal(0.0,0.1,self.n_factors)


    def fit(self,train_data):

        self.alpha = float(
            np.mean([r for (_, _, r) in train_data])
        )

        data  = list(train_data)

        for epoch in range(self.n_epochs):

            if self.shuffle:
                self.rng.shuffle(data)
            

            for u,i,r in data:
                # init p_u and q_i for cold start 
                self._init_item(i)
                self._init_user(u)

                pu = self.P[u]
                qi = self.Q[i]

                pred = self.alpha + self.beta_U[u] + self.beta_I[i] + np.dot(pu,qi)

                err = r - pred 

                # update alpha

                self.alpha += self.lr * err

                # save copies of pu, qi
                pu_old = pu.copy()
                qi_old = qi.copy()

                # update beta_u, beta_i

                self.beta_I[i] += self.lr * (err - self.reg*self.beta_I[i])
                self.beta_U[u] += self.lr * (err - self.reg*self.beta_U[u])

                # update pu , qi

                self.P[u] = self.lr * (err * qi_old - self.reg * pu_old)
                self.Q[i] = self.lr * (err*pu_old - self.reg*qi_old)

            mse = np.mean([(r - self.predict(u, i))**2 for u, i, r in data])
            print(f"Epoch {epoch+1}/{self.n_epochs}, train MSE = {mse:.4f}")


    def predict(self,u,i):
        # cold start handling 
        bu = self.beta_U[u] if u in self.beta_U else 0.0
        bi = self.beta_I[i] if i in self.beta_I else 0.0

        pu = self.P[u] if u in self.P else np.zeros(self.n_factors)
        qi = self.Q[i] if i in self.Q else np.zeros(self.n_factors)

        return self.alpha + bu + bi + float(np.dot(pu, qi))

        