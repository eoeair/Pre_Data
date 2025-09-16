import os
import numpy as np

folder_path = "data/Skeleton"
num_frames = []
num_bodies = []

# 遍历文件夹下的所有txt文件
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as f:
            lines = f.readlines()
            if len(lines) >= 2:
                try:
                    num_frame = int(lines[0].strip())
                    num_body = int(lines[1].strip())
                    num_frames.append(num_frame)
                    num_bodies.append(num_body)
                except ValueError:
                    print(f"Warning: {filename} contains non-integer in first two lines")

num_frames = np.array(num_frames)
num_bodies = np.array(num_bodies)

# 统计指标
print("num_frame statistics:")
print(f"  Max: {num_frames.max()}")
print(f"  99% percentile: {np.percentile(num_frames, 99)}")
print(f"  Mean: {num_frames.mean():.2f}")
print(f"  Median: {np.median(num_frames)}")

print("\nnum_body statistics:")
print(f"  Max: {num_bodies.max()}")
print(f"  99% percentile: {np.percentile(num_bodies, 99)}")
print(f"  Mean: {num_bodies.mean():.2f}")
print(f"  Median: {np.median(num_bodies)}")
