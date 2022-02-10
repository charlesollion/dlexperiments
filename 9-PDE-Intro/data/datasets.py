import jax.numpy as np
from jax import random

class BurgersDataset():
    def __init__(self, key, u0, batch_size=32, batch_f_size=512, N_u=100, N_f=5000):
        """
        Generate Burgers Dataset
        u0: border function at t=0
        """
        self.batch_size = batch_size
        self.batch_f_size = batch_f_size
        self.N_u = N_u
        self.N_f = N_f
        self.curr_idx = 0
        self.curr_f_idx = 0
        self.u0 = u0
        self.generate_data(key, N_u)


    def generate_data(self, key, N_u):
        # Generate border data
        key, subkey = random.split(key)
        data_type = random.uniform(key, (N_u,))>0.5
        key, subkey1, subkey2, subkey3 = random.split(key, 4)
        self.x_data = data_type * (random.uniform(subkey1, (N_u,))*2.0-1.0)+ \
            (1-data_type) * ((random.uniform(subkey2, (N_u,))>0.5)*2.0-1.0)
        self.t_data = data_type * 0 + ((1-data_type) * random.uniform(subkey3, (N_u,)))
        self.u_data = data_type * self.u0(self.x_data)
        # Generate inside data
        key1, key2 = random.split(key, 2)
        self.t_f_data = random.uniform(key1, (self.N_f, 1))
        self.x_f_data = random.uniform(key2, (self.N_f, 1)) * 2.0 - 1.0


    def border_batch(self):
        bstart = self.curr_idx * self.batch_size
        bend = (self.curr_idx + 1) * self.batch_size
        if bend >= self.N_u:
            bend = self.N_u-1
            self.curr_idx = 0
        else:
            self.curr_idx = self.curr_idx + 1
        x_ = np.expand_dims(self.x_data[bstart:bend], -1)
        t_ = np.expand_dims(self.t_data[bstart:bend], -1)
        u_ = np.expand_dims(self.u_data[bstart:bend], -1)
        return t_, x_, u_


    def inside_batch(self):
        bstart = self.curr_f_idx
        bend = bstart + self.batch_f_size
        if bend >= self.N_f:
            bend = self.N_f-1
            self.curr_f_idx = 0
        else:
            self.curr_f_idx = self.curr_f_idx + self.batch_f_size
        t_b = self.t_f_data[bstart:bend]
        x_b = self.x_f_data[bstart:bend]

        return t_b, x_b
