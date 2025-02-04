import os
import psutil
import numpy as np
from tqdm import tqdm
from shutil import copyfile
from numpy.lib.format import open_memmap

import jax
import jax.numpy as jnp

from functools import partial

@partial(jax.jit,static_argnames=('window_length', 'step', 'fs'))
def window_rfft(signal, window_length=100, step=20, fs=100):
    """
    Computes the RFFT for segments of signals using a sliding window.
    """
    n_frames, frame_length = signal.shape
    n_subframes = 1 + (frame_length - window_length) // step
    
    starts = jnp.arange(n_subframes) * step
    def get_window(frame, start):
        return jax.lax.dynamic_slice(frame, (start,), (window_length,))
    signal = jax.vmap(
        jax.vmap(get_window, in_axes=(None, 0)),  
        in_axes=(0, None)                         
    )(signal, starts)
    
    signal = signal - jnp.mean(signal, axis=2, keepdims=True)
    signal = jnp.abs(jnp.fft.rfft(signal))
    signal /= (fs / 2)
    signal = signal.at[...,0].set(signal[..., 0]/2)
    return signal.reshape(n_frames, -1)

def gen_rfft(dataset, shape):
    source = jnp.load('data/npy_data/{}/data.npy'.format(dataset),mmap_mode='r')
    data = open_memmap('data/fft_data/{}/data.npy'.format(dataset),dtype=np.float16,mode='w+',shape=shape)
    for i in range(shape[0]):
        
        for j in tqdm(range(shape[1])):
            data[i].__setitem__(j,window_rfft(source[i,j]))
        
        #data[i] = jax.vmap(window_rfft_jnp)(source[i])
            
def gen_test(shape=(9, 92726, 1071)):
    if not os.path.exists('data/fft_data/test'):
        os.mkdir('data/fft_data/test')
    
    source = jnp.load('data/npy_data/test/data.npy',mmap_mode='r')
    data = open_memmap('data/fft_data/test/data.npy',dtype=np.float16,mode='w+',shape=shape)
    
    for i in tqdm(range(shape[0])):
        data.__setitem__(i,window_rfft(source[i]))
    
    #data = jax.vmap(window_rfft_jnp)(source)

if __name__ == '__main__':

    #形状对应：模态，数据源(九个通道代表Acc、Gyr、Mag的各xyz通道)，帧，500个样本
    shape_list = [(4, 9, 196072, 1071),(4, 9, 28789, 1071)]

    sets = ('train', 'valid')

    modal_list = ("Bag","Hand","Hips","Torso")

    if not os.path.exists('data/fft_data'):
        os.mkdir('data/fft_data')
        for set in sets:
            path=os.path.join('data/fft_data',set)
            if not os.path.exists(path):
                os.mkdir(path)
    
    # copy label
    for set in sets:
        copyfile('data/npy_data/{}/label.npy'.format(set),'data/fft_data/{}/label.npy'.format(set))
        print("processed {} Label".format(set))

    # gen test
    gen_test()
    print("processed test data")

    # gen_rfft
    for i,set in enumerate(sets):
        gen_rfft(set, shape_list[i])
        print("processed {}".format(set))
