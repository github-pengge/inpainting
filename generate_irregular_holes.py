import h5py
import numpy as np 
from scipy.ndimage import interpolation
from scipy.ndimage.interpolation import rotate
from scipy.misc import imsave


'''
A way to generate irregular holes by sampling multiple rectangles. It's not optimized.
It's hard to obatin a unified measure for HEIGHT_WIDTH_RANGE and HEIGHT_WIDTH_PROB, so if you would like to 
create datasets with irregular holes of multiple resolutions, you might need to alter the parameters and 
create each resolution alone, and then merge them.
'''


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


################## Parameters ##################
size = (512, 512)
NUM = 20000
MAX_RECT = 80  # max number of rectangles

MAX_MASKEDS = int(size[0]*size[1]*0.6)  # max masked elements

RECT_MIN_MOVE = 10  # min movement
RECT_MAX_MOVE = 502  # max movement

ROTATE_ANGLE_RANGE = range(0, 180, 5)  # rectangle rotation angle range, len(ROTATE_ANGLE_RANGE)=len(ROTATE_ANGLE_PROB)
ROTATE_ANGLE_PROB = [2]*6 + [1]*24 + [2]*6  # 0-30, 30-150, 150-180
_sum = sum(ROTATE_ANGLE_PROB)
ROTATE_ANGLE_PROB = [p/_sum for p in ROTATE_ANGLE_PROB]  # normalize

HEIGHT_WIDTH_RANGE = range(10, 100)  # range of height and width, len(HEIGHT_WIDTH_RANGE)=len(PROB)
HEIGHT_WIDTH_PROB = [1]*30 + [3]*10 + [1.5]*30 + [1]*20  # 10-40, 40-50, 50-80, 80-100
_sum = sum(HEIGHT_WIDTH_PROB)
HEIGHT_WIDTH_PROB = [p/_sum for p in HEIGHT_WIDTH_PROB]  # normalize
################## Parameters ##################

stat = {ratio/10.: 0 for ratio in range(1, 11)}
with h5py.File("irregular_holes.hdf5", "w") as f:
    dset = f.create_dataset("{}x{}".format(*size), (NUM,1,*size), dtype=np.float32)
    for i in range(NUM):
        box = np.ones(size, dtype=np.float32)
        tmp = None
        for n in range(np.random.randint(2, MAX_RECT)):
            tmp = box.copy()
            h, w = np.random.choice(HEIGHT_WIDTH_RANGE, p=HEIGHT_WIDTH_PROB, size=2)
            rect = Rect(h, w, 0, 0)
            x, y = np.random.randint(RECT_MIN_MOVE, RECT_MAX_MOVE, size=2)
            rect = rect.move(x, y)
            box = rect.put_in(box)
            angle = np.random.choice(ROTATE_ANGLE_RANGE, p=ROTATE_ANGLE_PROB)
            box = rotate(box, angle, cval=1, reshape=False)
            if np.sum(box<=0.1) > MAX_MASKEDS:
                box = tmp
                break
        box[box>0.1] = 1
        box[box<=0.1] = 0
        dset[i, 0] = box
        ratio = int(10*(1-np.sum(box)/size[0]/size[1])+0.999999) / 10.
        stat[ratio] += 1
        if i % 500 == 0:
            print('{}/{}'.format(i, NUM))
        
print('Stats: ')
for ratio in sorted(stat.keys()):
    print('%s: %.4f' % (ratio, stat[ratio]/float(NUM)))
