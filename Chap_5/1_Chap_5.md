# 初识numpy\\pandas\\matplotlib\\sklearn

## 前言

除开现在火热的深度学习，基于统计学背景的传统机器学习在很长的一段时间都占据着数据挖掘领域重要的地位，并且现在在很多新的深度学习模型中，也体现了相当多的统计机器学习的思想。
因此，对统计机器学习以及相关工具库的充分学习和理解，对于后续进一步的提升有着非常重要和关键的作用。

本章将会介绍统计机器学习领域Python最常使用到的四个工具库:numpy、pandas、matplotlib以及集成了大部分统计机器学习模型的sklearn工具库，来帮助大家对课程的学习。

## numpy介绍
之前有些例子中我们已经使用了numpy来进行最基础的数据的处理。相信大家对numpy的使用有了一些初步的理解，这里我再详细介绍一下numpy的知识点。
numpy作为python中最基础的数学工具库，在实际应用中的重要性不言而喻。numpy不单单承担了数组的角色，在很多情况下也被用来当做矩阵在Python中进行代数运算，和pytorch框架中的tensor互为转换，从而实现强大的进阶功能。

### 创建一个空numpy数组

首先我们尝试创建一个空numpy数组，并且看看数组的数据类型。
```python
import numpy as np
n = np.empty([3,4],dtype=int)
print(type(n))
print(n)
```
从结果可以看出，n的数据类型是numpy.ndarray，ndarray类型表示一个n维的数组，在python中多被当作一般数组、向量或者矩阵使用。

### 创建一个全为1\全为0的数组

有时候我们想要创建一个全0或者全1的数组，以此来进行某些代数的计算，我们可以试着用下面两个函数来直接生成。

```python
import numpy as np
n1 = np.ones([3,4],dtype=int)
print(n1)

n0 = np.zeros([3,4],dtype=int)
print(n0)
```

#### 按照目标数组的形状创建形状相同的全1\全0的数组

假如已经有了一个数组n，需要再生成一个跟n形状相同的全为0或者全为1的数组，我们可以尝试采用如下方式：

```python
import numpy as np
n = np.empty([3,4],dtype=int)

print(np.zeros_like(n))
print(np.ones_like(n))
```

### 按照数值范围创建数组

```python
import numpy as np
n = np.arange(1,10,2,dtype=int)
#创建等差数组
n1 = np.linspace(1,20,5)
#创建等比数组
n2 = np.logspace(1,100,30)

```

### 多维数组切片

```python
import numpy as np
n = np.array([[1,2,3,4],[5,6,7,8],[1,3,5,7]])

print(n[1:,:])
print(n[:,:1])
```

### 数组形状操作

```python
import numpy as np
n = np.arange(10)
print(n.reshape(5,2))
```

除开reshape之外，我们也可以使用flatten()来专门将多维数组展平成一维数组，如下面例子所示。其中可以手动设置按照列还是按照行进行展平：‘C’是按行，‘F’是按列。

```python
import numpy as np
n = np.arange(10).reshape(2,5)
n1 = n.flatten(order = 'C')
n2 = n.flatten(order = 'F')
print(n1)
print(n2)
```

transpose方法用来交换数组的维度，如下面的例子所示。原先的维度按照0开始往后进行索引，然后transpose()中将索引手动进行重排，来达到想要的效果。

```python
import numpy as np
n = np.arange(10).reshape(1,2,5)
print(n)
n1 = n.transpose(1,0,2)
print(n1)
n2 = np.arange(6).reshape(2,3)
print(n2)
print(n2.T)
```

squeeze()方法是压缩数组中长度为1的维度，来实现数组的降维。

```python
import numpy as np
n = np.arange(10).reshape(1,2,5)
print(n,n.shape())
n1 = n.squeeze()
print(n1,n1.shape())
```

stack()方法表示将多个相同形状的数组进行叠加，hstack()和vstack()是它的衍生方法，表示按照行或者列进行叠加

```python
import numpy as np
n1 = np.arange(3)
n2 = np.arange(3)
n3 = np.arange(3)
print(np.stack((n1,n2),0))
print(np.hstack((n1,n2,n3)))
print(np.vstack((n1,n2,n3)))
```

### numpy线性代数计算

np.dot()表示矩阵点乘，matmul()表示1矩阵乘积，inner()表示矩阵内积。

```python
import numpy as np
n = np.arange(9).reshape(3,3)
m = np.arange(10,20,1).reshape(3,3)
print(np.dot(n,m))
print(np.matmul(n,m))
print(np.inner(n,m))
```

## pandas介绍

### Pandas Series

首先是pandas最基础的数据存储结构Series，在定义列表n过后通过pandas转成Series格式，可以得到以下结果。左边一列表示索引，可以通过手动来定义，右边一列是我们确定好的列表中的值。

```python
import pandas as pd
n = [1,2,3]
s = pd.Series(n)
print(s)
s1 =pd.Series(n,index=['a','b','c'])
print(s1)
#结果1：
0    1
1    2
2    3
dtype: int64
#结果2：
a    1
b    2
c    3
dtype: int64
```

### Pandas DataFrame

DataFrame是pandas中最常见的数据储存结构，可以把它当作无数个Series拼在一起组合成的一个整体。

```python
import pandas as pd

data = [['Google', 10], ['Runoob', 12], ['Wiki', 13]]

# 创建DataFrame
df = pd.DataFrame(data, columns=['Site', 'Age'])

# 使用astype方法设置每列的数据类型
df['Site'] = df['Site'].astype(str)
df['Age'] = df['Age'].astype(float)

print(df)
#结果：
     Site   Age
0  Google  10.0
1  Runoob  12.0
2    Wiki  13.0
```

```python
import pandas as pd

data = {'Site':['Google', 'Runoob', 'Wiki'], 'Age':[10, 12, 13]}

df = pd.DataFrame(data)

print (df)
```

我们也可以将多维数组转成DataFrame，设置好对应的列名即可：

```python
import numpy as np
import pandas as pd

# 创建一个包含网站和年龄的二维ndarray
ndarray_data = np.array([
    ['Google', 10],
    ['Runoob', 12],
    ['Wiki', 13]
])

# 使用DataFrame构造函数创建数据帧
df = pd.DataFrame(ndarray_data, columns=['Site', 'Age'])

# 打印数据帧
print(df)
#结果：
     Site Age
0  Google  10
1  Runoob  12
2    Wiki  13
```

我们也可以将定义好的多个Series组合成DataFrame：

```python
# 从 Series 创建 DataFrame
s1 = pd.Series(['Alice', 'Bob', 'Charlie'])
s2 = pd.Series([25, 30, 35])
s3 = pd.Series(['New York', 'Los Angeles', 'Chicago'])
df = pd.DataFrame({'Name': s1, 'Age': s2, 'City': s3})
```

```python
# 索引和切片
print(df[['Name', 'Age']])  # 提取多列
print(df[1:3])               # 切片行
print(df.loc[:, 'Name'])     # 提取单列
print(df.loc[1:2, ['Name', 'Age']])  # 标签索引提取指定行列
print(df.iloc[:, 1:])        # 位置索引提取指定列
```

### 数据清洗

有时候一个现成的DataFrame里面的数据可能会缺失(Nan)、格式错误或者数据超出范围，这时我们在进行数据分析之前要先处理一遍异常数据，进行数据清洗。

```python
DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)#清洗空值
```

```python
DataFrame.duplicated()#找出重复数据
DataFrame.drop_duplicates(inplace = True)#删除重复数据
```

## Matplotlib

matplotlib是一个非常重要的数据可视化的包，这里我以一个例子介绍一下他的基础功能。

```python
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
x = np.linspace(0, 10, 100)  # 生成一个包含100个点的x轴数据
y_line = np.random.randn(100)  # 生成100个随机数作为拆线图的y轴数据
# 拆线图
plt.figure(figsize=(10, 4))
plt.plot(x, y_line)
plt.title('Line Chart')
plt.show()
x_scatter = np.random.rand(50) * 10  # 生成50个随机数作为散点图的x轴数据
y_scatter = np.random.rand(50) * 10  # 生成50个随机数作为散点图的y轴数据
# 散点图
plt.figure(figsize=(6, 4))
plt.scatter(x_scatter, y_scatter)
plt.title('Scatter Plot')

x_bar = ['Category 1', 'Category 2', 'Category 3', 'Category 4']  # 条形图的类别
y_bar = np.random.rand(4) * 10  # 生成4个随机数作为条形图的高度数据
# 条形图
plt.figure(figsize=(8, 4))
plt.bar(x_bar, y_bar)
plt.title('Bar Chart')

labels_pie = ['A', 'B', 'C', 'D']  # 饼图的标签
sizes_pie = np.random.rand(4)  # 生成4个随机数作为饼图的数据
# 创建饼图
plt.figure(figsize=(6, 6))
plt.pie(sizes_pie, labels=labels_pie, autopct='%1.1f%%')
plt.title('Pie Chart')
```

上面例举了如何生成折线图、散点图、条形图以及饼图的相关例子，大家可以对照参考。

## Scikit-learn

sklearn全名是scikit-learn,是基于统计机器学习理论开发出来的一套数据挖掘软件库，里面基本上涵盖了从有监督学习到无监督学习的大部分模型，可以应对绝大多数情况下的数据挖掘任务。

由于sklearn的体系过于庞大，具体的使用细则这里无法展开讲，具体可以参考sklearn的官方文档：[sklearn官方文档](https://scikit-learn.org/stable/)

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据集，diabetes数据集是sklearn库中自带的数据集，可以直接加载使用
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# 使用前三个特征进行预测
diabetes_X = diabetes_X[:, :3]

# 划分训练集和测试集
diabetes_X_train, diabetes_y_train= diabetes_X[:-20],diabetes_y[:-20]
diabetes_X_test,diabetes_y_test = diabetes_X[-20:],diabetes_y[-20:]

# 创建线性回归模型
regr = linear_model.LinearRegression()

# 模型训练
regr.fit(diabetes_X_train, diabetes_y_train)

# 预测
diabetes_y_pred = regr.predict(diabetes_X_test)

print(diabetes_y_pred)
# 输出线性模型系数
print("Coefficients: \n", regr.coef_)
# MSE
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
```



