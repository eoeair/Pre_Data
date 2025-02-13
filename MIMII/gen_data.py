import asyncio
import os
from tqdm.asyncio import tqdm_asyncio
import numpy as np
import soundfile as sf
import psutil

async def main():
    # 结果存放列表
    file_list = []
    
    # 定义目录列表等
    snrs = ['-6dB', '0dB', '6dB']
    devices = ['fan', 'pump', 'slider', 'valve']
    ids = ['id_00', 'id_02', 'id_04', 'id_06']
    labels = ['abnormal', 'normal']
    
    # 定义映射字典
    snrm = {'-6dB': 1, '0dB': 2, '6dB': 3}
    devicem = {'fan': 1, 'pump': 2, 'slider': 3, 'valve': 4}
    labelm = {'abnormal': -1, 'normal': 1}
    
    # 遍历目录获取文件列表
    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.endswith('.wav'):
                path = os.path.relpath(root, 'data')
                parts = path.split(os.sep)
                if len(parts) == 4:
                    snr, device, _, label = parts
                    file_list.append((
                        os.path.join(root, file),
                        snrm[snr],
                        devicem[device],
                        labelm[label]
                    ))

    # 创建mmap文件
    data = np.lib.format.open_memmap('data/data.npy', dtype=np.float32, mode='w+', shape=(3 * 18019, 160000))
    label = np.lib.format.open_memmap('data/label.npy', dtype=np.int8, mode='w+', shape=(3 * 18019, 3))

    # 控制并发量
    sem = asyncio.Semaphore(psutil.cpu_count() * 3)

    async def process_file(i, file):
        async with sem:
            loop = asyncio.get_running_loop()
            # 异步执行文件读取和数据处理
            try:
                audio_data, _ = await loop.run_in_executor(
                    None, 
                    lambda: sf.read(file[0], always_2d=True)
                )
                # 写入mmap文件
                data[i] = audio_data.transpose(1, 0)[0].astype(np.float32)
                label[i] = [file[1], file[2], file[3]]
            except Exception as e:
                print(f"Error processing {file[0]}: {str(e)}")

    # 使用tqdm显示
    tasks = [process_file(i, file) for i, file in enumerate(file_list)]
    await tqdm_asyncio.gather(*tasks, desc="Processing files")

    print("processed data and label")

if __name__ == '__main__':
    asyncio.run(main())