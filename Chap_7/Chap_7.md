# Pytorch框架与计算机视觉工具库（OpenCV）介绍

## 前言

在深度学习领域目前主流的开发框架主要有两种，一种是Google开发的tensorflow，主要用于工业界进行工程开发；另一种是目前开源的pytorch深度学习框架，虽然没有Google的tensorflow那样便捷，但是由于其良好的开源生态，在学术界非常受欢迎。

所以本章计划对pytorch框架和计算机视觉OpenCV图像处理工具库进行一些简要的介绍

## Pytorch框架

[Pytorch官网](https://pytorch.org/) [Pytorch使用文档](https://pytorch.org/docs/stable/index.html)

由于pytorch的功能非常繁多，这里以建立一个多层感知器预测sklearn中糖尿病数据集为例，给大家展示一下pytorch的使用。

### 搭建简易的神经网络

```python
import sklearn.datasets as datasets
import torch.nn as nn
import torch.optim
import torch.utils.data as data

#获取datasets数据
feature,label = datasets.load_diabetes(return_X_y=True)#featrue(442,10),lable(442)
feature = torch.tensor(feature,dtype=torch.float32)
label = torch.tensor(label,dtype=torch.float32).unsqueeze(1)

#建立数据迭代器
def load_array(data_array,batch_size,is_Training=True):
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset,batch_size,shuffle=is_Training)
data_iter = load_array((feature[:-42,:],label[:-42,:]),100)

#建立全连接网络模型
class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.Linear = nn.Linear(10,512)
        self.relu = nn.ReLU()
        self.Linear2 = nn.Linear(512,512)
        self.relu1 = nn.ReLU()
        self.Linear3 = nn.Linear(512,1)
        self.Linear.weight.data.normal_(0,1)
        self.Linear.bias.data.fill_(0)
        self.Linear2.weight.data.normal_(0,1)
        self.Linear2.bias.data.fill_(0)
        self.Linear3.weight.data.normal_(0, 1)
        self.Linear3.bias.data.fill_(0)
    def forward(self,feature):
        f1 = self.Linear(feature)
        f2 = self.Linear2(self.relu(f1))
        f3 = self.Linear3(self.relu1(f2))
        return f3
#模型2
#net = nn.Sequential(nn.Linear(10,1))
#net[0].weight.data.normal_(0,1)
#net[0].bias.data.fill_(0)
#开始训练：
net = model()
#设置学习率和epoch：
lr = 0.0001
epochs = 10000
Loss = nn.MSELoss()
train = torch.optim.SGD(net.parameters(),lr=lr)

for epoch in range(epochs):
    for X,y in data_iter:
        train.zero_grad()
        l=Loss(net(X),y)
        l.sum().backward()
        train.step()
    l=Loss(net(feature[-42:,:]),label[-42:])
    print(f"{epoch+1}:Loss is {l}")
```

上面这个模型学习的效果虽然非常差，但是模型构建的方式都是一样的，有兴趣的同学可以下来自己接着上面的代码调参试试看。

## OpenCV工具库介绍

OpenCV是专门针对计算机图像处理所设计的开源库，里面对图像处理，图像变换以及图像与图像之间的融合有着专门的工具库来进行处理，OpenCV随着近几年来CV的发展，实现的功能越来越多但同时也变得繁杂。因此这里只介绍最基础的OpenCV的功能。

[OpenCV中文文档](https://opencv.apachecn.org/4.0.0/2.1-tutorial_py_image_display/)

### 常用图像处理方法

##### 图像读取

OpenCV通过imread()方法来进行图像数据的读取，第一个参数是读取图像的所在路径，之后的可选参数根据实际情况进行调整。

```python
import numpy as np
import cv2 as cv

＃加载彩色灰度图像
img = cv.imread('messi5.jpg'，0)
```

##### 图像显示

OpenCV通过imshow()进行阶段性显示，waitKey表示让图像暂停多久，0表示无限暂停。

```python
cv.imshow('image'，img）
cv.waitKey(0)
cv.destroyAllWindows()
```

##### 图像保存

imwrite()可以将图像写入指定路径，并且对图像文件命名

```python
cv.imwrite('messigray.png',img)
```

##### 访问图像像素值

imread()图像读完后就是个numpy矩阵，可以按照访问numpy的方式来访问图像像素。

```python
import numpy as np
import CV2
img = CV2.imread("img/yellow.jpg")
h,w,c = img.shape
#图像大小为128*128*3
print(h,w,c)
128 128 3
```

##### 图像简单变换

###### 平移变换

```python
import numpy as np
import cv2 as cv
img = cv.imread('messi5.jpg',0)
rows,cols = img.shape
M = np.float32([[1,0,100],[0,1,50]])
dst = cv.warpAffine(img,M,(cols,rows))
cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAllWindows()
```

###### 仿射变换

```python
img = cv.imread('drawing.png')
rows,cols,ch = img.shape
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv.getAffineTransform(pts1,pts2)
dst = cv.warpAffine(img,M,(cols,rows))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
```
