## 流程

1. 放置数据集：将数据下载到`data`，这一操作后，您的目录结构应该是
```
data
├───ECG5000_TEST.arff
└───ECG5000_TRAIN.arff
```
2. 数据集处理为csv格式：运行`process.ipynb`得到mfcc特征数据
3. 最终你会得到如下所展示的目录结构与文件
```
data
├───ECG5000_TEST.arff
├───train.npy
└───test.npy
```