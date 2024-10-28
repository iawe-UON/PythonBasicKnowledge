# Class，function和匿名函数

## 前言

这一章作为基础篇的最后一章，难度要比之前的第一章和第二章大上一些，也更加的抽象。但与此同时，这一张的知识进一步地体现了计算机思维在实际问题中的应用，也是我们对后续进阶知识掌握的一个关键跳板

## python函数（function）

首先是python函数。虽然现在python的各种开源包都有封装好的函数可以直接调用，但是实际问题中的复杂情况远远超出开源包中封装函数的处理范围。因此掌握如何设计一个自己的python函数是有必要的。并且在程序设计中，函数是程序设计和算法的核心，甚至有一门学科叫做函数式编程，所以函数的重要程度可见一斑。

### 函数定义

首先是如何定义一个函数，可以参考下面的例子：

```python
def maxn(a,b):
	if a>b:
        return a
    else:
        return b
```

以上是定义一个判断两个数输出最大值的一个函数。可以看到，定义一个函数首先要以 **def** 开头，然后设置自定义的函数名和圆括号 **()** ,圆括号内列出想要往函数内传递的参数（上面例子参数定义是a和b）。然后使用上面定义好的要传递的参数，再写函数内部想要进行的操作。完成函数的定义。

在定义完函数后，我们尝试一下调用这个函数解决一个小问题。

```python
a,b = 2,3
ans = maxn(a,b)
print(ans)
```

以上流程就是调用函数的一般流程。

定义函数的简单规则如下：

- 函数代码块以 **def** 关键词开头，后接函数标识符名称和圆括号 **()**。

- 任何传入参数和自变量必须放在圆括号中间，圆括号之间可以用于定义参数。

- 函数的第一行语句可以选择性地使用文档字符串—用于存放函数说明。

- 函数内容以冒号 **:** 起始，并且缩进。

- **return [表达式]** 结束函数，选择性地返回一个值给调用方，不带表达式的 return 相当于返回 None。

### 函数的参数传递

#### 必需参数传递

python函数在定义后，调用函数时（）内的参数必须按照顺序全部传递进函数，漏传或者没有按照顺序传递会报错或者出现错误答案。

```python
def printstring(str1，str2):
   print (str1)
   return

printstring("1","2")
printstring("2","1")
printme()
```

#### 关键字参数

有时候为了更加直观，我们会把定义的函数中的参数当作关键字，进行关键字传参。这时不能漏传参数，但是传递参数的顺序可以不用按照函数中定义的顺序来传递，因为这时参数的传递已经有了明确的指向。

```python
def printNameandAge(name,age):
   print ("name:",name)
   print ("age:",age)
   return

#调用printNameandAge函数
printNameandAge(age=50, name="Name")
```

#### 默认参数

在定义函数的参数时我们可以尝试在声明参数阶段就对其进行赋值，这样在没有其他的特殊情况下，调用函数时我们只需要传入没有进行赋值的关键字参数即可。有时我们需要对默认参数进行改变，则在调用函数时直接对默认参数的关键字进行传参即可。

```python
def printNameandAge( name,age=35):
   print ("name:", name)
   print ("age:", age)
   return
 
#调用printNameandAge函数
printNameandAge(age=50,name="Name")
printinfo(name="Name")
```

#### 不定长参数

有时候我们并不知道要传入的参数究竟有多少元素，这时我们需要专门申明不定长参数来进行处理，这样每次传入不同长度的元素都可以被当做参数交由函数进行处理。

```python
def printNum(arg1,*vartuple):
   "打印任何传入的参数"
   print ("output:")
   print (arg1)
   print (vartuple)
 
printNum( 70, 60, 50 )
```

## python匿名函数

### 匿名函数特点

lambda 函数的语法只包含一个语句，使用**lambda**关键字进行声明，然后列出需要的参数，冒号后申明要进行计算的表达式，如以下所示：

```python
lambda [arg1 [,arg2,.....argn]]:expression
```

### 直接定义匿名函数

```python
# 可写函数说明
sum = lambda arg1, arg2: arg1 + arg2
 
# 调用sum函数
print ("相加后的值为 : ", sum( 10, 20 ))
print ("相加后的值为 : ", sum( 20, 20 ))
```

### 在函数中嵌套定义匿名函数

我们可以将匿名函数封装在一个函数内，这样可以使用同样的代码来创建多个匿名函数（这里一定程度上体现了面向对象的编程思想）。以下例子将匿名函数封装在 myfunc 函数中，通过传入不同的参数来创建不同的匿名函数：

```python
def myfunc(n):
  return lambda a : a * n
 
mydoubler = myfunc(2)
mytripler = myfunc(3)
 
print(mydoubler(11))
print(mytripler(11))
```

此外还有更为复杂的匿名函数使用说明，这里不再过多赘述 [lambda详细教程](https://www.runoob.com/python3/python-lambda.html)

## Class类

python中的class类主要是面向对象思想的体现，这里不做详细解释，仅仅简单介绍一下class的概念：用来描述具有相同的属性和方法的对象的集合。它定义了该集合中每个对象所共有的属性和方法。对象是类的实例。

#### 定义Class

```python
class ClassName:
    <statement-1>
    ...
    <statement-N>
```

##### 定义Class、进行简单的实例化并调用

下面例子我们定义了一个固定id的打印hello world的class类，类的名字取做MyClass，并且展示了如何调用类中的方法。

```python
class MyClass():
    i = 12345
    def f(self):
        return 'hello world'
 
# 实例化类
x = MyClass()
 
# 访问类的属性和方法
print("MyClass 类的属性 i 为：", x.i)
print("MyClass 类的方法 f 输出为：", x.f())
```

##### init初始化

类有一个名为 **__init__()** 的特殊方法（**构造方法**），该方法在类实例化时会自动调用。类定义了 __init__() 方法，类的实例化操作会自动调用 __init__() 方法。如以下例子：

```python
def __init__(self):
    self.data = []
```

init()方法同样也可以承担参数传递的作用，参数通过init() 传递到类的实例化操作上。

```python
class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart
x = Complex(3.0, -4.5)
print(x.r, x.i)   
```

##### self的含义

这里再讲一下init()中self的含义。类的方法与普通的函数只有一个特别的区别——它们必须有一个额外的第一个参数名称, 按照惯例它的名称是 self。

在 Python中，self 是一个惯用的名称，用于表示类的实例（对象）自身。它是一个指向实例的引用，使得类的方法能够访问和操作实例的属性。当你定义一个类，并在类中定义方法时，第一个参数通常被命名为 self，尽管你可以使用其他名称，但强烈建议使用 self，以保持代码的一致性和可读性。

```python
class MyClass():
    def __init__(self, value):
        self.value = value

    def display_value(self):
        print(self.value)

# 创建一个类的实例
obj = MyClass(42) 

# 调用实例的方法
obj.display_value()
```

在上面的例子中，self 是一个指向类实例的引用，它在 **__init__** 构造函数中用于初始化实例的属性，也在 **display_value** 方法中用于访问实例的属性。通过使用 self，你可以在类的方法中访问和操作实例的属性，从而实现类的行为。

#### Class的方法

```python
#类定义
class people():
    #定义基本属性
    name = ''
    age = 0
    __weight = 0
    #定义构造方法
    def __init__(self,n,a,w):
        self.name = n
        self.age = a
        self.__weight = w
    def speak(self):
        print("%s 说: 我 %d 岁。" %(self.name,self.age))
 
# 实例化类
p = people('Name',10,30)
p.speak()
```

在类的内部，使用 **def** 关键字来定义一个方法，与一般函数定义不同，类方法必须包含参数 self, 且为第一个参数，self 代表的是类的实例。

面向对象的编程是一个复杂而且庞大的知识体系，这里只是作为简单的介绍，后面还有很多技巧：继承，多态，重写以及设置私有方法那些大家可以下来自行了解

