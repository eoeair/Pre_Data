# Pre_Data

Data preparation code to provide consistent and high-performance processing

## dep
`numpy tqdm numba joblib scipy`

## dataset

1. UAV-Human: Skeleton  --> `N C T V M`
2. NTURGB-D：Skeleton [ST-GCN]  --> `N C T V M`
3. NTURGB-D：Skeleton [CTR-GCN]
4. NW-UCLA：Skeleton  --> `N C T V M`
5. MIMII：Audio  --> `[SNR[MFCC,device,label]]`
6. SHL-2024: seqence  --> `Modal Channel Num sample`
7. ECG5000: seqence
8. Kuairec
9. Tenrec

## Target needs

1. Mem：Up to 16GB
2. Time：Faster as well as possible

## Be care

1. Data preprocessing uses a lot of performance optimizations, the goal of which is trying to strike a reasonable balance between speed and demand
2. This project has a high demand for I/O to ensure that it works on a medium with high I/O capability
3. Some datasets have different preprocessing patterns in different projects, and to avoid ambiguity, I have indicated the source in "[]".
4. `N C T V M` is `Num Channel Frames Joint Body`