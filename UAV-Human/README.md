## 流程

1. 解压数据集：将`Skeleton.zip`在`data`目录下解压，这一操作会自然的创建出一个子目录`Skeleton`,也就是说，这一操作后，您的目录结构应该是
```
data
└───Skeleton
    ├───P000S00G10B10H10UC022000LC021000A000R0_08241716.txt
    ├───P000S00G10B10H10UC022000LC021000A001R0_08241716.txt
```
2. 超参数划分： 运行`python dataset_profile.py`得到数据集超参数
3. 数据集处理为npy格式(joint模态)：运行`python gen_joint.py`得到joint模态数据
4. 数据集处理出angle模态数据（可选）：运行`python gen_modal.py --modal angle`得到motion模态的数据
5. 数据集处理出reverse模态合并（可选）：运行`python gen_modal.py --modal reverse`得到合并模态的数据
6. 最终你会得到如下所展示的目录结构与文件
```
└───data/v1
    ├───train
        ├───P000S00G10B10H10UC022000LC021000A000R0_08241716.txt
        ├───P000S00G10B10H10UC022000LC021000A001R0_08241716.txt
        └───...
    ├───test
        ├───P000S00G10B10H10UC022000LC021000A000R0_08241716.txt
        ├───P000S00G10B10H10UC022000LC021000A001R0_08241716.txt
        └───...
    ├── train_label.npy
    ├── train_bone_motion.npy
    ├── train_bone.npy
    ├── train_joint_bone.npy
    ├── train_joint_motion.npy
    ├── train_joint.npy
    ├── test_label.npy
    ├── test_bone_motion.npy
    ├── test_bone.npy
    ├── test_joint_bone.npy
    ├── test_joint_motion.npy
    ├── test_joint.npy
```