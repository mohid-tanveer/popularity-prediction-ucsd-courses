import numpy as np
from collections import defaultdict

class baseline_model:
    def __init__(self, lamb, n_iter):
        self.lamb = lamb
        self.n_iter = n_iter

        # params
        self.alpha = 0.0
        self.beta_u = defaultdict(float)
        self.beta_i = defaultdict(float)



    def fit(self, train_data, global_avg):

        alpha = global_avg

        users_per_item = defaultdict(list)
        items_per_user = defaultdict(list)

        n = len(train_data)

        # u : the professor
        # i : the course code
        # r : the reccomendation percentage
        for u, i , r in train_data:
            items_per_user[u].append((i,r))
            users_per_item[i].append((u,r))

        # ALS
        for _ in range(self.n_iter):
            
            # update alpha 
            sum_a  = 0.0
            for u,i,r in train_data:
                sum_a += r - self.beta_u[u] -self.beta_i[i]
            self.alpha = sum_a / n


            # update beta_U
            for u,items in items_per_user.items():
                sum_u = 0.0
                for i,r in items:
                    sum_u += (r - self.alpha - self.beta_i[i])
            
                self.beta_u[u] = sum_u / (self.lamb + len(items) )
                
            for i,users in users_per_item.items():
                sum_i = 0.0
                for u,r in users:
                    sum_i += (r- self.alpha - self.beta_u[u])

                self.beta_i[i] = sum_i / (self.lamb + len(users)) 


    def predict(self,u,i):
        return self.alpha + self.beta_u[u] + self.beta_i[i]

