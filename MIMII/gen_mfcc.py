import psutil
from tqdm import tqdm
import audioflux as af
from joblib import Parallel , delayed

import numpy as np
from numpy.lib.format import open_memmap

if __name__ == '__main__':
    source = np.load('data/data.npy',mmap_mode='r')
    data = open_memmap('data/mfcc_data.npy',dtype=np.float32,mode='w+',shape=(3 * 18019,13,153))
    # for simplify, only use channel 0
    Parallel(n_jobs=psutil.cpu_count(logical=True), verbose=0)(
        delayed(lambda i: 
            data.__setitem__(i, af.mfcc(source[i])[0])
        )(i) for i in tqdm(range(source.shape[0]))
    )
    print("processed mfcc data")