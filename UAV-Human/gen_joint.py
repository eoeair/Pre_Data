"""
Script to process raw data and generate dataset's binary files:
    - .npy skeleton data files: np.array of shape B x C x V x T x M
    - .npy label files: (label: list[int])
"""
import os
import re
import glob
import psutil
import numpy as np
from tqdm import tqdm
from joblib import Parallel , delayed
from numpy.lib.format import open_memmap

from preprocess import pre_normalization

MAX_BODY_TRUE = 2
MAX_BODY_KINECT = 4
NUM_JOINT = 17
MAX_FRAME = 300

def read_xyz(file, max_body=MAX_BODY_KINECT, num_joint=NUM_JOINT):
    """
    读取 skeleton 文件，返回 (C, T, V, M)
    C: 3 (x,y,z)
    T: 时间帧数 (pad 到 MAX_FRAME)
    V: 关节数
    M: 人数 (筛选能量最大的 MAX_BODY_TRUE 个)
    """
    with open(file, 'r') as f:
        num_frame = int(f.readline())
        data = np.zeros((max_body, num_frame, num_joint, 3), dtype=np.float32)

        for t in range(num_frame):
            num_body = int(f.readline())
            for m in range(min(num_body, max_body)):
                _ = f.readline()  # body info line, 忽略
                num_joint_this = int(f.readline())
                for j in range(min(num_joint, num_joint_this)):
                    vals = list(map(float, f.readline().split()))
                    data[m, t, j, :] = vals[:3]  # 只取 (x,y,z)

    # -------- Step1: 计算每个人的能量，选出前 MAX_BODY_TRUE -------- #
    energy = np.zeros(max_body, dtype=np.float32)
    for m in range(max_body):
        person = data[m]  # (T,V,C)
        mask = person.sum(-1).sum(-1) != 0
        person_valid = person[mask]
        if len(person_valid) > 0:
            energy[m] = (
                person_valid[:, :, 0].std()
                + person_valid[:, :, 1].std()
                + person_valid[:, :, 2].std()
            )
    select_idx = energy.argsort()[::-1][:MAX_BODY_TRUE]
    data = data[select_idx, :MAX_FRAME, :, :]  # (M,T,V,C)

    # -------- Step2: pad 到 MAX_FRAME -------- #
    T = data.shape[1]
    if T < MAX_FRAME:
        pad_shape = (data.shape[0], MAX_FRAME - T, data.shape[2], data.shape[3])
        data = np.concatenate([data, np.zeros(pad_shape, dtype=np.float32)], axis=1)

    # -------- Step3: 修复空帧 -------- #
    for i_p, person in enumerate(data):
        if person.sum() == 0:
            continue
        # 如果开头全 0 → 用第一个有效段填充
        if person[0].sum() == 0:
            mask = (person.sum(-1).sum(-1) != 0)
            tmp = person[mask].copy()
            person[:] = 0
            person[:len(tmp)] = tmp
        # 如果中间有空帧 → 用循环重复填充
        for i_f, frame in enumerate(person):
            if frame.sum() == 0:
                if person[i_f:].sum() == 0:
                    rest = len(person) - i_f
                    num = int(np.ceil(rest / i_f))
                    pad = np.concatenate([person[:i_f] for _ in range(num)], axis=0)[:rest]
                    data[i_p, i_f:] = pad
                    break

    return data.transpose(3, 1, 2, 0)  # (C, T, V, M)


def gendata(data_path,split):

    out_path = data_path
    data_path = os.path.join(data_path, split)

    skeleton_filenames = [os.path.basename(f) for f in glob.glob(os.path.join(data_path, "**.txt"), recursive=True)]

    FILENAME_REGEX = r'P\d+S\d+G\d+B\d+H\d+UC\d+LC\d+A(\d+)R\d+_\d+'
    label = [int(re.match(FILENAME_REGEX, i).groups()[0]) for i in skeleton_filenames]
    np.save('{}/{}_label.npy'.format(out_path, split), label)

    sample_name = [os.path.join(data_path, i) for i in skeleton_filenames]

    data = open_memmap('{}/{}_joint.npy'.format(out_path, split),dtype='float32',mode='w+',shape=((len(sample_name), 3, MAX_FRAME, NUM_JOINT, MAX_BODY_TRUE)))
    
    Parallel(n_jobs=psutil.cpu_count(logical=False), verbose=0)(delayed(lambda i,s: data.__setitem__(i,read_xyz(s, max_body=MAX_BODY_KINECT, num_joint=NUM_JOINT)))(i,s) for i,s in enumerate(tqdm(sample_name)))

    # check no skeleton
    for i in range(len(data)):
        if np.all(data[i,:] == 0):
            print("{} {} has no skeleton".format(data_path, i))

    data = data.transpose(0, 4, 2, 3, 1)  # N, C, T, V, M  to  N, M, T, V, C

    # Center the human at origin
    # center_joint: body joint index indicating center of body
    # sub the center joint #1 (spine joint in ntu and neck joint in kinetics)'
    for i_s, skeleton in enumerate(data):
        if skeleton.sum() == 0:
            continue
        # skeleton[0][:, center_joint:center_joint+1, :].copy() | uav center_joint=1
        main_body_center = skeleton[0][:, 1:2, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            # reshape(T, V, 1) | uav T=300 v=17
            mask = (person.sum(-1) != 0).reshape(300, 17, 1)
            data[i_s, i_p] = (data[i_s, i_p] - main_body_center) * mask

    data = pre_normalization(data).transpose(0, 4, 2, 3, 1) # N, M, T, V, C to N, C, T, V, M
    
if __name__ == '__main__':

    path_list = ['data/v1','data/v2']
    part = ['train','test']
    
    processes = []
    for path in path_list:
        for p in part:
            gendata(path, p)