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
        self.generate_data(key)


    def generate_data(self, key):
        # Generate border data
        key, subkey = random.split(key)
        data_type = random.uniform(key, (self.N_u, 1))>0.5
        key, subkey1, subkey2, subkey3 = random.split(key, 4)
        self.x_data = data_type * (random.uniform(subkey1, (self.N_u, 1))*2.0-1.0)+ \
            (1-data_type) * ((random.uniform(subkey2, (self.N_u, 1))>0.5)*2.0-1.0)
        self.t_data = data_type * 0 + ((1-data_type) * random.uniform(subkey3, (self.N_u, 1)))
        self.u_data = data_type * np.expand_dims(self.u0(self.x_data), -1)
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
        x_ = self.x_data[bstart:bend]
        t_ = self.t_data[bstart:bend]
        u_ = self.u_data[bstart:bend]
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


class KPPDataset(BurgersDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def generate_data(self, key):
        # Generate border data
        key, subkey = random.split(key)
        #data_type = random.uniform(key, (self.N_u, 1))>0.5
        data_type = np.ones((self.N_u, 1))
        key, subkey1, subkey2, subkey3, subkey4, subkey5 = random.split(key, 6)
        x_data_t0 = data_type * (random.uniform(subkey1, (self.N_u, 2)))
        border_dim = random.uniform(subkey2, (self.N_u, 1))>0.5
        border_sign = random.uniform(subkey4, (self.N_u, 1))>0.5
        border_value = random.uniform(subkey5, (self.N_u, 1))
        x_data_borderx = (1-data_type) * border_dim * np.concatenate((border_sign, border_value), axis=-1)
        x_data_bordery = (1-data_type) * (1-border_dim) * np.concatenate((border_value, border_sign), axis=-1)
        self.x_data = x_data_t0 + x_data_borderx + x_data_bordery
        self.t_data = data_type * 0 + ((1-data_type) * random.uniform(subkey3, (self.N_u, 1)))
        self.u_data = data_type * np.expand_dims(self.u0(self.x_data), -1)
        # Generate inside data
        key1, key2 = random.split(key, 2)
        self.t_f_data = random.uniform(key1, (self.N_f, 1))
        self.x_f_data = random.uniform(key2, (self.N_f, 2))
