import numpy as np 
import h5py


class CelebA():
    def __init__(self, celeba_data_path, hole_data_path):
        self.resolution = ['4x4', '8x8', '16x16', '32x32', '64x64', '128x128', '256x256', '512x512', '1024x1024']
        self._base_key = 'data'
        self.celeba_data_path = celeba_data_path
        self.hole_data_path = hole_data_path
        self.celeba = h5py.File(celeba_data_path, 'r')
        self.holes = h5py.File(hole_data_path, 'r')
        self._data_len = {k:len(self.celeba[self._base_key+k]) for k in self.resolution}
        self._hole_len = {k:len(self.holes[k]) for k in self.resolution}
        assert all([self._base_key+resol in self.celeba.keys() for resol in self.resolution])

    def __call__(self, batch_size, size=512):
        key = '{}x{}'.format(size, size)
        assert key in self.resolution
        data_key = self._base_key + key
        data_idx = np.random.randint(self._data_len[key], size=batch_size)
        hole_idx = np.random.randint(self._hole_len[key], size=batch_size)
        x = np.array([self.celeba[data_key][i] for i in data_idx], dtype=np.float32)/255.0
        hole = np.array([self.holes[key][i] for i in hole_idx], dtype=np.float32)
        return x, hole

