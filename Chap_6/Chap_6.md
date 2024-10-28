# 统计机器学习实战 —— 以鸢尾花数据集分类为例

## 前言

今天我们使用前几章学习过的知识，特别是numpy，pandas以及sklearn这些库，来建立统计机器学习模型对鸢尾花数据进行分类预测。

## 模型建立思路

### 数据读取加载

首先是数据读取，由于iris数据集是在sklearn中封装好的现成的数据，所以我们直接调用即可。

```python
import numpy as np
import sklearn.datasets as datasets
import sklearn.svm as SVM
import pandas as pd

#加载数据集
index = np.arange(0,150)
np.random.shuffle(index)
feature,label = datasets.load_iris(return_X_y=True)
```

然后我们可以检查一下feature和lable的值，发现lable是有序的，这样不利于训练，所以我们需要使用index作为索引，并将其进行打乱来构建训练集和数据集：

```python
index_train = index[:-10]
index_test = index[-10:]
#划分训练集和测试集：
feature_train,lable_train = feature[index_train,:],label[index_train]
feature_test,lable_test = feature[index_test],label[index_test]
```

然后是选择模型，这里我们使用SVM来进行多分类任务

```python
#选择SVM模型
reg = SVM.SVC(decision_function_shape='ovo')
reg.fit(feature_train,lable_train)
lable_prediction=reg.predict(feature_test)
```

最后我们测试一下模型的准确性，结果发现准确度达到了90%，说明模型效果不错。

```python
for i in range(10):
    if lable_prediction[i]==lable_test[i]:
        print("True")
    else:
        print("False")
```



