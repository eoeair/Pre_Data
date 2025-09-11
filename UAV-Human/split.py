import os
import glob
import asyncio
import aiofiles
from tqdm import tqdm

# 限制最大并发数
sem = asyncio.Semaphore(64)

async def async_copyfile(src, dst):
    """异步拷贝文件，带信号量控制"""
    async with sem:  # 控制同时执行的任务数量
        async with aiofiles.open(src, 'rb') as fsrc, aiofiles.open(dst, 'wb') as fdst:
            await fdst.write(await fsrc.read())

def split_list(skeleton_filenames, train_list, all_path, path):
    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'test')

    tasks = []
    for basename in skeleton_filenames:
        pid = int(basename[1:4])  # 从文件名里取 ID
        filename = os.path.join(all_path, basename)
        if pid in train_list:
            target_filename = os.path.join(train_path, basename)
        else:
            target_filename = os.path.join(test_path, basename)
        tasks.append(async_copyfile(filename, target_filename))
    return tasks

async def main():
    # V1 和 V2 的训练集 ID
    train_v1_list = [0, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 59, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 73, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 98, 100, 102, 103, 105, 106, 110, 111, 112, 114, 115, 116, 117, 118]
    train_v2_list = [0, 3, 4, 5, 6, 8, 10, 11, 12, 14, 16, 18, 19, 20, 21, 22, 24, 26, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 49, 52, 54, 56, 57, 59, 60, 61, 62, 63, 64, 66, 67, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 83, 84, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118]
    lists = [train_v1_list, train_v2_list]
    path_list = ['data/Skeleton','data/v1', 'data/v2', 'data/v1/train', 'data/v1/test', 'data/v2/train', 'data/v2/test']
    
    # 确保目录存在
    for path in path_list:
        os.makedirs(path, exist_ok=True)

    # 获取所有 *.txt 文件
    skeleton_filenames = [os.path.basename(f)
        for f in glob.glob(os.path.join(path_list[0], "*.txt"), recursive=True)
    ]

    # 并发执行 v1 和 v2 的划分
    tasks = split_list(skeleton_filenames, lists[0], path_list[0], path_list[1]) + \
            split_list(skeleton_filenames, lists[1], path_list[0], path_list[2])

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await coro

if __name__ == '__main__':
    asyncio.run(main())