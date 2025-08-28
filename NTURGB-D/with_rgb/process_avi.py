import os
import cv2
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import newwork.feeder.augmentations as augmentations


def read_video_to_memory(video_path):
    """高效读取视频到内存"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return np.array(frames)


def process_joints_vectorized(frames, skeleton_data, joint_crops, crop_size):
    """向量化处理关节裁剪"""
    nframes = len(frames)
    frame_height, frame_width = frames.shape[1:3]
    half_size = crop_size // 2

    for i in range(nframes):
        num_bodies_in_frame = skeleton_data["nbodys"][i]

        for body_idx in range(min(num_bodies_in_frame, 4)):  # 最多处理4个身体
            if f"rgb_body{body_idx}" not in skeleton_data:
                continue

            # 获取关节坐标
            joints = skeleton_data[f"rgb_body{body_idx}"][i]

            # 处理每个关节
            for joint_idx in range(25):
                x, y = joints[joint_idx]
                if np.isnan(x) or np.isnan(y):
                    with open("nan_skel.txt", "a") as f:
                        f.write(f"{skeleton_data['file_name']}\n")
                    continue

                x, y = int(round(x)), int(round(y))

                # 计算裁剪区域的边界
                x_min = int(round(x - half_size))
                x_max = int(round(x + half_size))
                y_min = int(round(y - half_size))
                y_max = int(round(y + half_size))

                # 处理边界情况（超出视频范围时填充黑色）
                pad_left = max(0, -x_min)
                pad_right = max(0, x_max - frame_width)
                pad_top = max(0, -y_min)
                pad_bottom = max(0, y_max - frame_height)

                # 调整裁剪区域到有效范围
                x_min = max(0, x_min)
                x_max = min(frame_width, x_max)
                y_min = max(0, y_min)
                y_max = min(frame_height, y_max)

                # 提取有效区域
                crop = frames[i][y_min:y_max, x_min:x_max]

                # 如果需要填充（超出边界）
                if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                    crop = np.pad(
                        crop,
                        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )

                # 确保裁剪大小一致（可能由于奇数尺寸或填充不匹配）
                # if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                #     crop = cv2.resize(crop, (crop_size, crop_size))

                joint_crops[i, body_idx, joint_idx] = crop

                return joint_crops


def process_avi_file(file_name, avi_data_path, skel_data_path, crop_size, window_size, output_dir):
    """处理单个AVI文件"""
    try:
        # 1. 读取视频
        avi_path = os.path.join(avi_data_path, file_name + "_rgb.avi")
        frames = read_video_to_memory(avi_path)

        # 2. 加载骨骼数据
        skeleton_path = os.path.join(skel_data_path, file_name + ".skeleton.npy")
        skeleton_data = np.load(skeleton_path, allow_pickle=True).item()

        # 3. 检查帧数一致性
        nframes = len(frames)
        if nframes != skeleton_data["skel_body0"].shape[0]:
            print(f"Frame mismatch in {file_name}: video={nframes}, skeleton={skeleton_data['skel_body0'].shape[0]}")
            return

        # 4. 预分配内存
        joint_crops = np.zeros((nframes, 4, 25, crop_size, crop_size, 3), dtype=np.uint8)

        # 5. 处理关节
        joint_crops = process_joints_vectorized(frames, skeleton_data, joint_crops, crop_size)

        # 6. 后处理
        joint_crops = joint_crops.transpose(5, 0, 2, 3, 4, 1)
        valid = nframes
        joint_crops = augmentations.crop_subsequence_rgb(joint_crops, valid, window_size)

        # 7. 保存结果
        output_path = os.path.join(output_dir, file_name + ".npy")
        np.save(output_path, joint_crops)

    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")


def process_avi(data_root_path, avi_file_name, avi_data_path, skel_data_path, crop_size, window_size, output_dir):
    """主处理函数"""
    avi_file_name_path = os.path.join(data_root_path, avi_file_name)
    avi_data_path = os.path.join(data_root_path, avi_data_path)
    skel_data_path = os.path.join(data_root_path, skel_data_path)
    output_dir = os.path.join(data_root_path, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(avi_file_name_path, "r") as f:
        file_names = [line.strip() for line in f]

    # 使用更高效的并行处理
    Parallel(n_jobs=14, backend="multiprocessing")(
        delayed(process_avi_file)(file_name, avi_data_path, skel_data_path, crop_size, window_size, output_dir) for file_name in tqdm(file_names)
    )


if __name__ == "__main__":
    data_root_path = "/root/data1/chenpeidong"
    avi_file_name = "skes_available_name.txt"
    avi_data_path = "nturgb+d_rgb"
    skel_data_path = "nturgb+d_skeletons_npy"
    crop_size = 32
    window_size = 64
    output_dir = "nturgb+d_skeletons_crop"

    process_avi(data_root_path, avi_file_name, avi_data_path, skel_data_path, crop_size, window_size, output_dir)
