import psutil
from joblib import Parallel , delayed
from numpy.lib.format import open_memmap

from get_angle import getThridOrderRep

# uav graph
    # (10, 8), (8, 6), (9, 7), (7, 5), # arms
    # (15, 13), (13, 11), (16, 14), (14, 12), # legs
    # (11, 5), (12, 6), (11, 12), (5, 6), # torso
    # (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2) # nose, eyes and ears

graph = ((10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (11, 12), (5, 6), (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2))

# bone
#fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]

# motion  
#fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]

# angle ,From UAV-SAR
def gen_angle(dataset, set):
    print(dataset, set)
    data = open_memmap('./data/{}/{}_joint.npy'.format(dataset, set),mode='r')
    N, C, T, V, M = data.shape
    fp_sp = open_memmap('./data/{}/{}_angle.npy'.format(dataset, set),dtype='float32',mode='w+',shape=(N, 9, T, V, M))
    Parallel(n_jobs=psutil.cpu_count(logical=False), verbose=0)(delayed(lambda i: fp_sp.__setitem__(i,getThridOrderRep(data[i])))(i) for i in range(N))

# reverse , From FR-AGCN
#颠倒时间轴，反向排列
