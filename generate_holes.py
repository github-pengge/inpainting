import h5py
import numpy as np 
from scipy.ndimage import interpolation
from scipy.ndimage.interpolation import rotate


class Rect:
    def __init__(self, h, w, top_left_x=0, top_left_y=0):
        self.h = h 
        self.w = w 
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y

    def move(self, x, y):
        rect = Rect(self.h, self.w, self.top_left_x+x, self.top_left_y+y)
        return rect

    def put_in(self, box):
        tlx = (self.top_left_x+box.shape[0]) % box.shape[0]
        tly = (self.top_left_y+box.shape[1]) % box.shape[1]
        box[tlx:(tlx+self.w), tly:(tly+self.h)] = 0
        return box


sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
NUM = 12000
N = 2000
M = 2
with h5py.File("holes_hq.hdf5", "w") as f:
    for size in sizes:
        dset = f.create_dataset("{}x{}".format(size, size), (NUM,1,size,size), dtype=np.float32)

        rectangle_sizes = [size//2, size//4, size//3]

        for i, rect_size in enumerate(rectangle_sizes):
            rect = Rect(rect_size, rect_size, 0, 0)
            _from, _to = size//4, (size*3)//4
            for n in range(N):
                box = np.ones((size, size), dtype=np.float32)
                x, y = np.random.randint(_from, _to, size=2)
                new_rect = rect.move(x, y)
                box = new_rect.put_in(box)
                for m in range(M):
                    _do = True
                    while _do:
                        angle = np.random.randint(40)*9
                        rotated_box = rotate(box, angle, cval=1, reshape=False)
                        rotated_box[rotated_box>0.5] = 1
                        rotated_box[rotated_box<=0.5] = 0
                        if not np.all(rotated_box==0) and not np.all(rotated_box==1):
                            dset[i*(N*M)+M*n+m,0] = rotated_box
                            _do = False
                if n % 50 == 0:
                    print('{}x{}: {}/{}, {}/{}'.format(size, size, i, len(rectangle_sizes), n, N))
