import os
import psutil
from tqdm import tqdm
import soundfile as sf
from joblib import Parallel , delayed

import numpy as np
from numpy.lib.format import open_memmap

if __name__ == '__main__':
    # 结果存放列表
    file_list = []

    # 定义目录列表等
    snrs = ['-6dB', '0dB', '6dB']
    devices = ['fan', 'pump', 'slider', 'valve']
    ids = ['id_00', 'id_02', 'id_04', 'id_06']
    labels = ['abnormal','normal']

    # 定义映射字典
    snrm = {'-6dB': 1, '0dB': 2, '6dB': 3}
    devicem = {'fan': 1, 'pump': 2, 'slider': 3, 'valve': 4}
    labelm = {'abnormal': -1, 'normal': 1}
    
    for root, dirs, files in os.walk('data'):
        for file in files:
            # 如果是 wav 文件
            if file.endswith('.wav'):
                # 获取当前路径的相对部分
                path = os.path.relpath(root, 'data')
                # 假设路径结构是 snr/device/id/label
                parts = path.split(os.sep)

                if len(parts) == 4:  # 确保路径符合预期结构
                    snr, device, _, label = parts
                    file_list.append((os.path.join(root, file), snrm[snr], devicem[device],labelm[label]))

    data = open_memmap('data/data.npy',dtype=np.float32,mode='w+',shape=(3 * 18019,160000))
    label = open_memmap('data/label.npy',dtype=np.int8,mode='w+',shape=(3 * 18019,3))
    # for simplify, only use channel 0
    Parallel(n_jobs=psutil.cpu_count(logical=True), verbose=0)(
        delayed(lambda i, file: (
            data.__setitem__(i, sf.read(file[0])[0].transpose(1,0)[0]),
            label.__setitem__(i, [file[1], file[2], file[3]])
        ))(i, file) for i, file in enumerate(tqdm(file_list))
    )
    print("processed data and label")