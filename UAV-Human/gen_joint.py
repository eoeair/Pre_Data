"""
Script to process raw data and generate dataset's binary files:
    - .npy skeleton data files: np.array of shape B x C x V x T x M
    - .npy label files: (label: list[int])
"""
import os
import re
import glob
import asyncio
import numpy as np
import tensorstore as ts

from joblib import Parallel, delayed
from preprocess import pre_normalization

def split(dataset_path):
    # V1 和 V2 的训练集 ID
    train_v1_list = [0, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 59, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 73, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 98, 100, 102, 103, 105, 106, 110, 111, 112, 114, 115, 116, 117, 118]
    train_v2_list = [0, 3, 4, 5, 6, 8, 10, 11, 12, 14, 16, 18, 19, 20, 21, 22, 24, 26, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 49, 52, 54, 56, 57, 59, 60, 61, 62, 63, 64, 66, 67, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 83, 84, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118]

    v1_train, v1_valid = [], []
    v2_train, v2_valid = [], []

    # 获取所有 *.txt 文件
    skeleton_filenames = [os.path.basename(f)
        for f in glob.glob(os.path.join(dataset_path, "*.txt"), recursive=True)
    ]

    for name in skeleton_filenames:
        pid = int(name[1:4])  # 从文件名里取 ID
        path = os.path.join(dataset_path, name)
        if pid in train_v1_list:
            v1_train.append(path)
        else:
            v1_valid.append(path)

        if pid in train_v2_list:
            v2_train.append(path)
        else:
            v2_valid.append(path)

    return v1_train, v1_valid, v2_train, v2_valid

def read_xyz(file):
    """
    # dataset-level param （be profiled before process）
    读取 skeleton 文件，返回 (M, T, V, C)
    M: 人数 (最大人数只有2人，不必筛选能量最大的 MAX_BODY_TRUE 个)
    T: 时间帧数 (MAX_FRAME = 300)
    V: 关节数
    C: 3 (x,y,z)
    """
    idx = 0
    with open(file, 'r') as f:
        lines = f.read().splitlines()
    num_frame = int(lines[idx])
    idx += 1

    data = np.zeros((2, 300, 17, 3), dtype=np.float32) # M T V C
    
    for t in range(min(num_frame,300)):
        num_body = int(lines[idx])
        idx += 1
        for m in range(num_body):
            idx += 1 # body info line, 忽略
            idx += 1 # num_joint line,忽略
            for j in range(17):
                vals = list(map(float, lines[idx].split()))
                data[m, t, j, :] = vals[:3]  # 只取 (x,y,z)
                idx += 1
    # -------- 修复开头空帧 -------- #
    frame_mask = data.sum(axis=(2,3)) != 0
    for m in range(2):
        valid_idx = np.where(frame_mask[m])[0]
        if valid_idx.size == 0:
            # 全部空，不处理
            continue
        # 开头空帧,用第一个有效帧覆盖
        first_valid, last_valid = valid_idx[0], valid_idx[-1]
        n_valid = last_valid - first_valid + 1
        if first_valid > 0:
            data[m, :n_valid] = data[m, first_valid:last_valid+1]
            data[m, n_valid:] = 0

    # 计算每帧是否全零 (M, T)
    frame_mask = data.sum(axis=(2,3)) != 0
    for m in range(2):
        valid_idx = np.where(frame_mask[m])[0]
        if valid_idx.size == 0:
            # 全部空，不处理
            continue

        # 循环重复填充末尾空帧
        last_valid = valid_idx[-1]
        if last_valid < 300 - 1:
            repeated_frames = data[m, :last_valid+1]
            n_repeat = int(np.ceil((300 - last_valid - 1) / repeated_frames.shape[0]))
            pad = np.tile(repeated_frames, (n_repeat,1,1))[:300 - last_valid - 1]
            data[m, last_valid+1:] = pad

    return data

def preprocess(data):
    # Center the human at origin
    # center_joint: body joint index indicating center of body
    # sub the center joint #1 (spine joint in ntu and neck joint in kinetics)
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

    return pre_normalization(data)

async def extract_x(file_list, split):
    x = await ts.open({
        'driver': 'n5',
        'kvstore': {'driver': 'file', 'path': os.path.join('data', split)},
        'metadata': {
            'compression': {'type': 'blosc', 'cname': 'zstd', 'clevel': 3, 'shuffle': 0},
            'dataType': 'float32',
            'dimensions': [len(file_list), 3, 300, 17, 2],
            'blockSize': [16, 3, 300, 17, 2],
        },
        'create': True, 
        'delete_existing': True,
    })
    array = np.array(Parallel(n_jobs=8)(delayed(read_xyz)(f) for f in file_list)) 
    array = preprocess(array)
    await x.write(array.transpose(0, 4, 2, 3, 1)) # N, M, T, V, C to N, C, T, V, M

async def extract_y(file_list, regex, split):
    y = await ts.open({
        'driver': 'n5',
        'kvstore': {'driver': 'file', 'path': os.path.join('data', split)},
        'metadata': {
            'compression': {'type': 'blosc', 'cname': 'zstd', 'clevel': 3, 'shuffle': 0},
            'dataType': 'int32',
            'dimensions': [len(file_list)],
        },
        'create': True, 
        'delete_existing': True,
    })
    label = [int(regex.match(os.path.basename(f)).groups()[0]) for f in file_list]
    await y.write(label)

async def extract():
    FILENAME_REGEX = r'P\d+S\d+G\d+B\d+H\d+UC\d+LC\d+A(\d+)R\d+_\d+'
    pattern = re.compile(FILENAME_REGEX)

    v1_train, v1_valid, v2_train, v2_valid = split('data/Skeleton')

    splits = {
        "v1_train": v1_train,
        "v1_valid": v1_valid,
        "v2_train": v2_train,
        "v2_valid": v2_valid,
    }

    for name, files in splits.items():
        print(f"Extracting {name} ...")
        await extract_x(files, f"{name}_x")
        await extract_y(files, regex=pattern, split=f"{name}_y")

if __name__ == '__main__':
    asyncio.run(extract())