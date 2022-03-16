[TOC]



# Pycharm同步项目到服务器需注意

**在图中的位置配置 mapping 映射的位置**，才能在选择 解释器后 将项目上传到你想要的位置

![image-20220309172802390](https://gitee.com/KangPeiLun/images/raw/master/images/image-20220309172802390.png)

#### 在服务器上用pip安装包时要在前面加上python -m

**这样才能将pip安装的包，安装到conda创建的虚拟环境中去，否则会把包安装到 (base)环境下**

```python
python -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

#### pycharm连接服务器后，代码运行过程中生成文件夹不会在本地自动生成，需要手动从服务器上拉取相关文件夹

![image-20220310134129083](https://gitee.com/KangPeiLun/images/raw/master/images/image-20220310134129083.png)



# 相关技巧

### conda导入/导出虚拟环境

```python
#导出当前环境 
conda env export > py3.6.yaml

# 导入环境 
conda env create -f py3.6.yaml
```

### Linux将程序挂后台运行

```python
nohup python train_cycle.py > log.txt 2>&1 &
# python train_cycle.py 为你的命令
# log.txt：将程序运行过程中的标准输出 全部存入 log.txt文件
```



# Pytorch注意点总结

> 1. pytorch中颜色通道是第一个维度，跟很多工具包都不一样，需要转换
>
> 2. pytorch中参与运算的浮点数必须是 FloatTensor 的格式
>
> 3. pytorch中默认图片的维度为 `[C, H, W]`  即 `[通道, 长度, 宽度]`
>
> 4. `nn.CrossEntropyLoss`中`已经实现了softmax功能`，因此在分类任务的最后一层fc后不需要加入softmax激活函数
>
> 5. pytorch中`nn.Linear`层`只是对输入数据的最后一个维度进行变换`
>
>     >   比如input:[batch, h, w, c] 那么在定义Linear时这样写 nn.Linear(c, output_size)，则经过Linear后的结果为 [batch, h, w, output_size]
>     >
>     >   比如input:[batch, h * w * c] 那么在定义Linear时这样写 nn.Linear(h * w * c, output_size)，则经过Linear后的结果为 [batch, output_size]
>     
> 6. pytorch中当用view对tensor的形状进行变换时，若提示出错，则需要用`tensor.contiguous()`将数据变为`在内存上连续`，然后才能使用view

# 深度学习知识点

## 1.卷积核

在进行卷积计算时，一个卷积核的 通道c 和 特征图的通道数 是一致的，计算时 **对应通道的特征图和卷积核对应的通道 做 内积运算**，`最后将该卷积核对不同通道计算的值相加再加上偏置项`，就可以得到新的特征图

PS：**一个特征图可以理解为一个通道**

<img src="https://gitee.com/KangPeiLun/images/raw/master/images/image-20220114134924755.png" alt="image-20220114134924755" style="zoom: 67%;" />

### 卷积结果计算公式

![image-20220114150724625](https://gitee.com/KangPeiLun/images/raw/master/images/image-20220114150724625.png)



## 2.池化层(压缩，下采样)——**只会改变特征图长和宽，不改变通道数**

**池化层不涉及任何计算**，一般pooling之后，特征图的长宽会变为原来的一半，整体变为原来的四分之一

`使用pooling会损失一些特征信息，因此在pooling之后的下一个Conv中将特征图 翻倍，以弥补丢失的信息`

![image-20220114141008529](https://gitee.com/KangPeiLun/images/raw/master/images/image-20220114141008529.png)

## 3.只有带可训练参数的结构才能称为一层

下图中 Conv，FC层带可训练的参数，故一共有7层

而Relu，Pool不带可训练参数，故不能称之为层

<img src="https://gitee.com/KangPeiLun/images/raw/master/images/image-20220114141639527.png" alt="image-20220114141639527" style="zoom:67%;" />

## 4.感受野——越大越好

第一个Conv对应特征图 是根据 上一层特征图 的 整个大小计算而来的，即 First Conv中红色部分的3x3特征图 是根据 Input的5x5计算来的；Second Conv中红色部分是根据First Conv的3x3计算来的

我们把网络结构中 能够感受到的最大的 特征图的尺寸，作为整个网络的 感受野；如下图中的 感受野为 5x5

**网络越到后面，特征越抽象，而且每个特征所涉及的感受野也更大了**

<img src="https://gitee.com/KangPeiLun/images/raw/master/images/image-20220114143743300.png" alt="image-20220114143743300" style="zoom:67%;" />

## 5.inception网络结构

可以简单的理解为 横向的拓展网络，从多种尺寸从图像中提取特征

**网络有多个分支，不同分支同时训练，然后在汇集点进行特征融合**

[Inception网络模型解析](https://www.cnblogs.com/dengshunge/p/10808191.html)

<img src="https://gitee.com/KangPeiLun/images/raw/master/images/1456303-20190504162301047-898768446.png" alt="1456303-20190504162301047-898768446" style="zoom: 80%;" />

<img src="https://gitee.com/KangPeiLun/images/raw/master/images/1456303-20190504162327115-2119269106.png" alt="1456303-20190504162327115-2119269106" style="zoom:80%;" />

> - a）采用**不同大小的卷积核意味着不同大小的感受野，最后拼接意味着不同尺度特征的融合**； 
> - b）之所以卷积核大小采用1、3和5，主要是为了方便对齐；
> - c）文章说很多地方都表明pooling挺有效，所以Inception里面也嵌入了；
> - d）网络越到后面，特征越抽象，而且每个特征所涉及的感受野也更大了，因此随着层数的增加，3x3和5x5卷积的比例也要增加

# Pytorch编程技巧

## 1.使用map做数据类型的转换

**map()** 会根据提供的函数对指定序列做映射。

第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。

```python
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)   # 使用map方法，可以将数据全部映射为对应的数据类型
)
```

这里的torch.tensor可以看作是一个function，即function可以为要转换的数据类型



## 2.torch.max()的用法

```python
output = torch.max(input, dim)

# 输入
# input是softmax函数输出的一个tensor
# dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值

# 输出
# 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
```

## 3.assert 断言的用法

```python
total_layers = len([param for param in model.parameters()])  # 记录该网络共有多少层的参数可以学习
assert (freeze_layer < total_layers), f'the total_layers is {total_layers}, but freeze_layer is {freeze_layer}'

# 只有当 assert 后面括号中的 判断语句 为 False 时，程序才会终止，才会出现你写的报错信息
```

## 4.nn.CrossEntropyLoss()等价于nn.LogSoftmax()+nn.NLLLoss()

如果最后一层已经LogSoftmax()了，就不能nn.CrossEntropyLoss()来计算了

因为**nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合**

## 5.加载Pytorch内置模型时也需要分配device

```python
# PS: 注意 在加载 pytorch内置的模型时，也需要将模型分配一边device，否则会报错
# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0
model = model.to(config.device)
```

## 6.dataloader辨析

```python
len(dataloader.dataset)  # 获取整个数据集的数量
len(dataloader)  # 获取一共有多少个batch
```

## 7.Image.thumbnail和Image.resize的区别

> - `resize()`函数会返回一个`Image`对象, `thumbnail()`函数返回`None`。
> - `resize()`中的`size`参数直接设定了`resize`之后图片的规格, 而`thumbnail()`中的`size`参数则是设定了`x/y`上的最大值. 也就是说, 经过`resize()`处理的图片可能会被拉伸, 而经过`thumbnail()`**处理的图片不会被拉伸**。
> - `thumbnail()`函数内部调用了`resize()`, 可以认为`thumbnail()`是对`resize()`的一种封装。
> - `resize()`可以使图片变大或变小，`thumbnail()`**只能使图片缩小**

`thumbnail()`可以理解为**成比例的缩小**

## 8.函数类型注释

```python
def add(x:int, y:int) -> int:
    pass

# 这样的写法叫类型注释，用于辅助说明形参的数据类型，以及范围值的数据类型
# (x:int, y:int) 表示说明 形参数据类型
# -> int 表示 说明返回值数据类型

类型注释和普通的注释一样，只是起到一个说明的作用，不会对代码造成功能性的影响
```

## 9.torchvision.datasets.ImageFolder用法

ImageFolder要求数据按一下格式存放，文件夹的名字就是该类图片的 类别名

![image-20220115215239117](https://gitee.com/KangPeiLun/images/raw/master/images/image-20220115215239117.png)

![image-20220115215641282](https://gitee.com/KangPeiLun/images/raw/master/images/image-20220115215641282.png)

```python
# 定义用于训练和测试的数据增强方法
'''
    一定要保证训练集和测试集 的数据处理方法是一致的。测试集可以不包含训练集中的 数据增强部分
'''
data_transformer = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),  # 随机旋转，-45到45度之间
        transforms.CenterCrop(224),     # 从中心开始裁剪, 保留一个224x224大小的图像，相当于直接原图像进行裁剪了
        transforms.RandomHorizontalFlip(p=0.5),  # 以50%的概率随机 水平 翻转
        transforms.RandomVerticalFlip(p=0.5),   # 以50%的概率随机 垂直 翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),     # 以25%概率对图像进行灰度化，转换后的图片还是 3 通道的
        transforms.ToTensor(),  # 必须把数据转换为tensor格式
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 三个通道对应的均值，标准差，这些值是由resnet在对应数据集上训练时所计算得到的均值和标准差，用他们已经计算好的值效果会好些
    ]),
    'valid': transforms.Compose([
        transforms.Resize((256, 256)), # 切记 Resize操作中的参数 必须是两个数(num1, num2)
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # 必须把数据转换为tensor格式
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 标准化，三个通道对应的均值，标准差
    ])
}


image_dataset = datasets.ImageFolder(os.path.join(config.data_dir, x), data_transformer['train'])
# 第一个参数为所有类别所在的文件夹路径，第二个参数为对这些数据应用的变换

image_dataset.classes	# 获取该数据集所有的类别有哪些
```

## 10.把torch.tensor转化成numpy.array有下面几种方法

#### 如果tensor是放在cpu上，那么很简单一个命令搞定：

```python
import torch as t
a = t.tensor([1,2,3])
b = a.numpy()
print(b)
# output: array([1, 2, 3])
```

#### 如果tensor是放在GPU上，那么在用.numpy()之前先得用.cpu()把它传到cpu上。如下：

```python
import torch as t
a = t.tensor([1,2,3]).cuda()

b = a.cpu().numpy()

print(b)
# output: array([1, 2, 3])
```

#### 如果tensor**本身还包含梯度信息**，需要先把用.detach()梯度信息剔除掉，再转成numpy。如下：

```python
import torch as t
a = t.tensor([1,2,3]).cuda()
net = ... # 建立一个神经网络
out= net(a)  # 该输出是带有梯度的

b = out.cpu().detach().numpy()  # !!!

print(b)
# output: array([1, 2, 3])
```

## 11.把numpy.array转化成torch.tensor有下面几种方法

### t.Tensor()或者t.tensor()或者t.as_tensor(). 他们功能基本一致，可互换使用

t.Tensor()和t.tensor()的区别是 torch.Tensor是torch.tensor 和 torch.empty的集合. 为了避免歧义和困惑，才把 torch.Tensor 分拆成了torch.tensor 和`torch.empty. 所以t.tensor和t.Tensor()可以互换，没有谁更好。不过推荐使用t.tensor()，就类似list()，np.array()一样

而**t.tensor()和t.as_tensor()的区别是t.tensor()会复制原来的数据**，`t.as_tensor()会尽力不复制，尽力尝试与原来的numpy.array共享内存`


### 可使用t.from_numpy()，与原Numpy数据共享内存

上文中说过的t.Tensor()会复制数据，那么同时，数据的格式也变了。t.Tensor()是t.FloatTensor()的另一个名字，他们会把数据转化成32位浮点数。即使原来的numpy.array数据是int或者float64,都会被转化成32位浮点数

而使用**t.from_numpy()的话就会保留数据的格式**，这是因为t.from_numpy()会共用同一个内存，不会复制数据

>`.Tensor()`和`.tensor()`(复制数据),`.as_tensor()`(尽量不复制数据), `.from_numpy()`(不复制数据，共享内存)

## 12.torch中clone()与detach()操作

### .clone

>- 返回一个新的tensor，这个tensor与原始tensor的数据不共享一个内存(也就是说， **两者不是同一个数据，修改一个另一个不会变**)。
>- requires_grad属性与原始tensor相同，若requires_grad=True，计算梯度，但不会保留梯度，梯度会与原始tensor的梯度相加。

### .detach()

>- 返回一个新的tensor，这个tensor与原始tensor的数据共享一个内存(也就是说，**两者是同一个数据，修改原始tensor，new tensor也会变； 修改new tensor，原始tensor也会变**)。
>- require_grad设置为False（也就是网上说的从计算图中剥除,不计算梯度）

 

`PS：通常 .clone() 和 .detach() 会一起使用`

```python
    image = tensor.to('cpu').clone().detach()
    # 首先将GPU上的数据挪到CPU上，然后复制一个新的tensor(不共享内存，但会包含原来的梯度)，最后再从这个新的tensor中得到不含梯度的tensor
    image = image.numpy().squeeze()
```

## 13.Numpy中的squeeze()函数

**作用**：从数组的形状中删除单维度条目，即**把shape中为1的维度去掉**，`只要为1的维都会被删掉`

如 a = (1, 10, 1)  使用squeeze()后变为 a=(10,)

```python
numpy.squeeze(a,axis = None)
'''
    1）a表示输入的数组； 
    2）axis用于指定需要删除的维度，但是指定的维度必须为单维度，否则将会报错；
    3）axis的取值可为None 或 int 或 tuple of ints, 可选。若axis为空，则删除所有单维度的条目；
    4）返回值：数组
    5) 不会修改原数组；
'''

# 例子
import numpy as np

a  = np.arange(10).reshape(1,10)
# array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])  
a.shape # (1,10)

b = np.squeeze(a)  # 这里将shape中为1的维度给去掉了
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
b.shape # (10,)
```

## 14.Pytorch模型断点续练方法

### 两种保存模型的方法

> #### 1.直接保存，然后加载
>
> ````python
> torch.save(model, path) # 直接保存整个模型
> 
> model = torch.load(path) # 直接加载模型
> ````
>
> #### 2.保存模型的参数，然后加载（`推荐写法`）
>
> ```python
> torch.save(model.state_dict(), path) # 保存模型的参数
> 
> model = Model()                         # 先初始化一个模型
> model.load_state_dict(torch.load(path)) # 再加载模型参数，加载模型不用再赋值
> ```
>
> `PS：官方推荐第二种方法`**第一种方法容易出错**



详见：[PyTorch实现断点继续训练](https://zhuanlan.zhihu.com/p/133250753)

将网络训练过程中的网络的权重，优化器的权重保存，以及epoch 保存，便于继续训练恢复

在训练过程中，可以根据自己的需要，每多少代，或者多少epoch保存一次网络参数，便于恢复，提高程序的鲁棒性。

`如果训练过程中 learning_rate 是变化的，则还需要保存lr_scheduler`

```python
optimizer = optim.Adam(model.parameters(), lr=config.learn_rate)  # 默认使用Adam优化器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, scheduler_gamma=config.scheduler_gamma)

checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer":optimizer.state_dict(),
    	"lr_scheduler": lr_schedule.state_dict(),
        "epoch": epoch,   # 训练结束时的epoch
    	"acc": best_acc,  # 训练结束时模型的准确率
    }
    if not os.path.isdir("./models/checkpoint"):
        os.mkdir("./models/checkpoint")
    torch.save(checkpoint, './models/checkpoint/ckpt_best_%s.pth' %(str(epoch)))
```

### 加载恢复，断点续练

```python
#加载恢复
if RESUME:
    path_checkpoint = "./model_parameter/test/ckpt_best_50.pth"  # 断点路径
    checkpoint = torch.load(path_checkpoint)  # 加载断点

    model.load_state_dict(checkpoint['state_dict'])  # 加载模型可学习参数
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch']  # 设置开始的epoch
    lr_schedule.load_state_dict(checkpoint['lr_schedule'])#加载lr_scheduler
    
# 断点续练
for epoch in range(start_epoch+1,80):  # ！！！注意这里是如何恢复epoch的

    optimizer.zero_grad()

    optimizer.step()
    lr_schedule.step()


    if epoch %10 ==0:
        print('epoch:',epoch)
        print('learning rate:',optimizer.state_dict()['param_groups'][0]['lr'])
        checkpoint = {
            "state_dict": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
            'lr_schedule': lr_schedule.state_dict()
        }
        if not os.path.isdir("./model_parameter/test"):
            os.mkdir("./model_parameter/test")
        torch.save(checkpoint, './model_parameter/test/ckpt_best_%s.pth' % (str(epoch)))
```

## 15.argparse 命令行选项、参数和子命令解析器

详见：[argparse --- 命令行选项、参数和子命令解析器 — Python 3.10.2 文档](https://docs.python.org/zh-cn/3/library/argparse.html)

```python
# 整个argparse的使用主要有三部分组成
import argparse              #导入argparse模块

parser = argparse.ArgumentParser(description='')  #定义一个将命令行字符串解析为Python对象的对象

parser.add_argument()   	 #指定程序能接收哪些命令行选项

opt = parser.parse_args()    #从特定的命令行中返回一些数据
```

## 16.pytorch中优化器optimizer.param_groups[0]的含义

> - optimizer.param_groups：是长度为2的list，其中的元素是2个字典； 
> - optimizer.param_groups[0]：长度为6的字典，包括 [‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’] 这6个参数
> - optimizer.param_groups[1]：好像是表示优化器的状态的一个字典

## 17.torch.optim.lr_scheduler调整学习率用法

`torch.optim.lr_scheduler`模块提供了一些根据epoch训练次数来调整学习率（learning rate）的方法。一般情况下我们会设置随着epoch的增大而逐渐减小学习率从而达到更好的训练效果

> **scheduler.step（）** 按照Pytorch的定义是用来 **更新优化器的学习率**的，一般是按照epoch为单位进行更换，即多少个epoch后更换一次学习率，因而`scheduler.step()放在epoch这个大循环下`

### 常见的学习率调整策略有几种

#### LambdaLR 自定义调整学习率

将每个参数组的学习率设置为初始lr与给定函数的乘积，计算公式是

**new_lr = base_lr \* lmbda(self.last_epoch)**

```python
#函数原型
torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)

#使用方法
optimizer = optim.SGD([{'params': net.features.parameters()}, # 默认lr是1e-5
                       {'params': net.classifiter.parameters(), 'lr': 1e-2, "momentum" :0.9,                
                       "weight_decay" :1e-4}],
                      lr=1e-3)
lambda1 = lambda epoch: epoch // 30
lambda2 = lambda epoch: 0.95 ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
```

#### MultiplicativeLR

将每个参数组的学习率乘以指定函数中给定的系数，计算公式是：

**group[‘lr’] \* lmbda(self.last_epoch)**

```python
#函数原型
torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)
#使用方法
lmbda = lambda epoch: 0.95
scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
```

#### StepLR 等间隔调整

每隔多少个epoch，学习率衰减为原来的gamma倍

```python
#函数原型
torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False)
#使用方法
# Assuming optimizer uses lr = 0.5 for all groups
# lr = 0.5     if epoch < 30
# lr = 0.05    if 30 <= epoch < 60
# lr = 0.005   if 60 <= epoch < 90
# ...
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
```

#### ExponentialLR 指数衰减调整

每个参数组的学习率按照gamma曲线每个epoch衰减一次

```python
torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1, verbose=False)
```

#### ReduceLROnPlateau 自适应调整学习率

该策略能够读取模型的性能指标，当该指标停止改善时，持续关系几个epochs之后，自动减小学习率。

```python
torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)

# mode（min or max):min指示指标不再减小时降低学习率，max指示指标不再增加时，降低学习率
# factor: new_lr = lr * factor Default: 0.1.
# patience:　观察几个epoch之后降低学习率　Default: 10.

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min')
```

#### CosineAnnealingLR 余弦退火衰减

以余弦函数为周期，并在每个周期最大值时重新设置学习率。以初始学习率为最大学习率，以 2∗Tmax2*Tmax2∗Tmax 为周期，在一个周期内先下降，后上升。

```python
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)

# T_max(int)	一次学习率周期的迭代次数，即 T_max 个 epoch 之后重新设置学习率。
# eta_min(float)	最小学习率，即在一个周期中，学习率最小会下降到 eta_min，默认值为 0。
# last_epoch	最后一个EPOCH 默认-1，可不设置
```

#### CosineAnnealingWarmRestarts 

跟余弦退火类似，只是在学习率上升时使用热启动。这个 Scheduler 在各种比赛中也经常用到

详见：[余弦AnnealingWarmRestarts — PyTorch 1.10.1 文档](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)



## 18.技巧：np.clip用法

np.clip是一个截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值。

```python
np.clip(a, a_min, a_max, out=None)

# a：输入矩阵；
# a_min：被限定的最小值，所有比a_min小的数都会强制变为a_min；
# a_max：被限定的最大值，所有比a_max大的数都会强制变为a_max；
# out：可以指定输出矩阵的对象，shape与a相同

# 例
x= np.arange(12)
print(np.clip(x,3,8))
# [3 3 3 3 4 5 6 7 8 8 8 8]
# 可以看到，小于3的数都变为了3，大于8的数都变为了8
```

## 19.技巧：pytorch中squeeze()和unsqueeze()函数介绍

### unsqueeze(dim) 增加一个值为1的维度

**dim表示在哪个维度上增加**

> 1. 首先初始化一个a，可以看出a的维度为（2，3）
>
>    ![20180812155855509](https://gitee.com/KangPeiLun/images/raw/master/images/20180812155855509.png)
>
> 2. 在第二维增加一个维度，使其维度变为（2，1，3）
>
>    ![20180812160119403](https://gitee.com/KangPeiLun/images/raw/master/images/20180812160119403.png)
>
> 
>
> ​	可以看出a的维度已经变为（2，1，3）了，同样如果需要在倒数第二个维度上增加一个维度，那么使用b.unsqueeze(-2)

### squeeze() 去掉值为1的维度

> 1. 首先得到一个维度为（1，2，3）的tensor
>
>    ![20180812160833709](https://gitee.com/KangPeiLun/images/raw/master/images/20180812160833709.png)
>
> 2. 使用squeeze()函数将第一维去掉，维度已经变为（2，3）
>
> ```python
> tensor.squeeze()  # 将tensor中所有值为1的维度去掉
> ```

## 20.技巧：TensorboardX可视化训练过程

[tensorboardX 官方文档](https://tensorboardx.readthedocs.io/en/latest/tensorboard.html)

详见：[详解PyTorch项目使用TensorboardX进行训练可视化](https://blog.csdn.net/bigbennyguo/article/details/87956434)

Tensorboard 是 TensorFlow 的一个附加工具，可以记录训练过程的数字、图像等内容，以方便研究人员观察神经网络训练过程。可是对于 PyTorch 等其他神经网络训练框架并没有功能像 Tensorboard 一样全面的类似工具，一些已有的工具功能有限或使用起来比较困难 (tensorboard_logger, visdom 等) 。TensorboardX 这个工具使得 TensorFlow 外的其他神经网络框架也可以使用到 Tensorboard 的便捷功能。

### Tips

> 1. 如果在进入 embedding 可视化界面时卡住，请更新 tensorboard 至最新版本 (>=1.12.0)。
> 2. tensorboard 有缓存，如果进行了一些 run 文件夹的删除操作，最好**重启** tensorboard，以避免无效数据干扰展示效果。
> 3. 如果执行 `add` 操作后没有实时在网页可视化界面看到效果，试试**重启** tensorboard

### 基本用法

首先，需要创建一个 SummaryWriter 的示例：

```python
from tensorboardX import SummaryWriter

# Creates writer1 object.
# The log will be saved in 'runs/exp'
writer1 = SummaryWriter('runs/exp')

# Creates writer2 object with auto generated file name
# The log directory will be something like 'runs/Aug20-17-20-33'
writer2 = SummaryWriter()

# Creates writer3 object with auto generated file name, the comment will be appended to the filename.
# The log directory will be something like 'runs/Aug20-17-20-33-resnet'
writer3 = SummaryWriter(comment='resnet')


# ======================================================
# 代码中常见用法
writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
for epoch in range(epochs):
    writer.add_scalar("loss/train", loss.item(), total_batch)
    writer.add_scalar("loss/dev", dev_loss, total_batch)
    writer.add_scalar("acc/train", train_acc, total_batch)
    writer.add_scalar("acc/dev", dev_acc, total_batch)
```

以上展示了三种初始化 SummaryWriter 的方法：

1. 提供一个路径，将使用该路径来保存日志
2. 无参数，默认将使用 `runs/日期时间` 路径来保存日志
3. 提供一个 comment 参数，将使用 `runs/日期时间-comment` 路径来保存日志

一般来讲，我们对于每次实验新建一个路径不同的 SummaryWriter，也叫一个 **run**，如 `runs/exp1`、`runs/exp2`。

接下来，我们就可以调用 SummaryWriter 实例的各种 `add_something` 方法向日志中写入不同类型的数据了。想要在浏览器中查看可视化这些数据，只要在命令行中开启 tensorboard 即可：

```python
# 在cmd中进入logdir的父级目录中，然后运行一下命令
tensorboard --logdir=<your_log_dir>
```

其中的 `<your_log_dir>` 既**可以是单个 run 的路径**，如上面 writer1 生成的 `runs/exp`；也**可以是多个 run 的父目录<不同的实验结果会被叠加到一个图上面>**，如 `runs/` 下面可能会有很多的子文件夹，每个文件夹都代表了一次实验，我们令 `--logdir=runs/` 就可以在 tensorboard 可视化界面中方便地横向比较 `runs/` 下不同次实验所得数据的差异。



### 在pytorch中使用tensorboardX网页无法访问

```python
将浏览器中的网址修改为

http://localhost:6006/
        或
http://127.0.0.1:6006/
```

### add_scalar 记录数字常量

```python
add_scalar(tag, scalar_value, global_step=None, walltime=None)
```

参数

- **tag** (string): 数据名称，不同名称的数据使用不同曲线展示
- **scalar_value** (float): 数字常量值
- **global_step** (int, optional): 训练的 step
- **walltime** (float, optional): 记录发生的时间，默认为 `time.time()`

需要注意，这里的 `scalar_value` 一定是 float 类型，如果是 PyTorch scalar tensor，则需要调用 `.item()` 方法获取其数值。我们一般会使用 `add_scalar` 方法来记录训练过程的 **loss、accuracy、learning rate** 等数值的变化，直观地监控训练过程。

```python
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/scalar_example')
for i in range(10):
    writer.add_scalar('quadratic', i**2, global_step=i)
    writer.add_scalar('exponential', 2**i, global_step=i)
```

这里，我们在一个路径为 `runs/scalar_example` 的 run 中分别写入了二次函数数据 `quadratic` 和指数函数数据 `exponential`，在浏览器可视化界面中效果如下：

<img src="https://gitee.com/KangPeiLun/images/raw/master/images/20190228173235116.png" alt="20190228173235116" style="zoom: 50%;" />

### add_image 记录单个图像数据的变化过程，注意需要 pillow 库的支持

```python
add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
```

`add_image 方法只能一次插入一张图片`

参数

- **tag** (string): 数据名称
- **img_tensor** (torch.Tensor / numpy.array): 图像数据
- **global_step** (int, optional): 训练的 step
- **walltime** (float, optional): 记录发生的时间，默认为 `time.time()`
- **dataformats** (string, optional): 图像数据的格式，默认为 `'CHW'`，即 `Channel x Height x Width`，还可以是 `'CHW'`、`'HWC'` 或 `'HW'` 等

我们一般会使用 `add_image` 来**实时观察生成式模型的生成效果**，或者**可视化分割、目标检测的结果，帮助调试模型**。

```python
from tensorboardX import SummaryWriter
import cv2 as cv

writer = SummaryWriter('runs/image_example')
for i in range(1, 6):
    writer.add_image('countdown',
                     cv.cvtColor(cv.imread('{}.jpg'.format(i)), cv.COLOR_BGR2RGB),
                     global_step=i,
                     dataformats='HWC')
```

`add_image` 写入记录。这里我们使用 opencv 读入图片，opencv 读入的图片通道排列是 BGR，因此需要先转成 RGB 以保证颜色正确，并且 `dataformats` 设为 `'HWC'`，而非默认的 `'CHW'`。调用这个方法一定要保证数据的格式正确，像 PyTorch Tensor 的格式就是默认的 `'CHW'`。效果如下，可以拖动滑动条来查看不同 `global_step` 下的图片：

![20190228203954868](https://gitee.com/KangPeiLun/images/raw/master/images/20190228203954868.gif)



### histogram 记录一组数据的直方图

```python
add_histogram(tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None)
```

参数

- **tag** (string): 数据名称
- **values** (torch.Tensor, numpy.array, or string/blobname): 用来构建直方图的数据
- **global_step** (int, optional): 训练的 step
- **bins** (string, optional): 取值有 ‘tensorflow’、‘auto’、‘fd’ 等, 该参数决定了分桶的方式，详见[这里](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html)。
- **walltime** (float, optional): 记录发生的时间，默认为 `time.time()`
- **max_bins** (int, optional): 最大分桶数

我们可以通过**观察数据、训练参数、特征的直方图**，**了解到它们大致的分布情况**，辅助神经网络的训练过程

```python
from tensorboardX import SummaryWriter
import numpy as np

writer = SummaryWriter('runs/embedding_example')
writer.add_histogram('normal_centered', np.random.normal(0, 1, 1000), global_step=1)
writer.add_histogram('normal_centered', np.random.normal(0, 2, 1000), global_step=50)
writer.add_histogram('normal_centered', np.random.normal(0, 3, 1000), global_step=100)
```

使用 `numpy` 从不同方差的正态分布中进行采样。打开浏览器可视化界面后，我们会发现多出了 "DISTRIBUTIONS" 和 "HISTOGRAMS" 两栏，它们都是用来观察数据分布的。其中在 "HISTOGRAMS" 中，同一数据不同 step 时候的直方图可以上下错位排布 (OFFSET) 也可重叠排布 (OVERLAY)。上下两图分别为 "DISTRIBUTIONS" 界面和 "HISTOGRAMS" 界面

![20190228211417803](https://gitee.com/KangPeiLun/images/raw/master/images/20190228211417803.png)

![20190228211434630](https://gitee.com/KangPeiLun/images/raw/master/images/20190228211434630.gif)



### graph 可视化一个神经网络

```python
add_graph(model, input_to_model=None, verbose=False, **kwargs)
```

参数

- **model** (torch.nn.Module): 待可视化的网络模型
- **input_to_model** (torch.Tensor or list of torch.Tensor, optional): 待输入神经网络的变量或一组变量  `input_to_model不能为None`

该方法可以可视化神经网络模型，TensorboardX 给出了一个[官方样例](https://github.com/lanpa/tensorboardX/blob/master/examples/demo_graph.py)大家可以尝试。样例运行效果如下：



![20190228213040835](https://gitee.com/KangPeiLun/images/raw/master/images/20190228213040835.gif)

## 21.Pytorch中transpose()和permute()区别

`transpose()`**只能一次操作两个维度**；`permute()`可以**一次操作多维数据**，且必须传入所有维度数，因为`permute()`的参数是`int*`。

```python
# transpose() 中传入要交换位置的两个维度的索引
x.transpose(0, 1)  等价于  x.transpose(1, 0)

# permute() 中需要传入数据所有的维度索引
假设y有三个维度
y.permute(0, 1, 2)  # 这就就表示不对y的维度进行变化
y.permute(1, 2, 0)  # 表示将y的原来第一个维度 挪到 最后一个维度，
```

## 22.技巧：nn.Linear()可以看作是一个矩阵乘法，且该矩阵可以被训练得到

```python
self.W_Q = nn.Linear(config.d_model, config.d_q * config.n_heads, bias=False)  # q,k的维度必须相同，否则无法点积
self.W_K = nn.Linear(config.d_model, config.d_k * config.n_heads, bias=False)
self.W_V = nn.Linear(config.d_model, config.d_v * config.n_heads, bias=False)

'''
	在Attention机制中，需要将 输入的数据 乘上 权重矩阵 得到对应的 Q，K，V
	而 nn.Linear() 可以看作是一个矩阵相乘的操作，只需要将 bias 设置为False 即可
'''
```

## 23.pytorch中的矩阵乘法

详见：[随笔1: PyTorch中矩阵乘法总结 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/100069938)

### 二维矩阵乘法 torch.mm() **不支持broadcast操作**

```python
torch. mm(mat1, mat2, out = None)

其中 mat1(n x m) , mat2(m x d)，输出out的维度是(n x d)。该函数torch.mm()一般只用来计算两个二维矩阵的矩阵乘法，而且 #不支持broadcast操作
```

### 三维带Batch矩阵乘法 torch.bmm() **不支持broadcast操作**

```python
torch.bmm(bmat1, bmat2, out = None)

由于神经网络训练一般采用mini-batch，经常输入的是三维带batch矩阵，所以提供torch.bmm(bmat1, bmat2, out = None) ,其中bmat1(B × n x m) ,bmat2(B x m x d)，输出 out的维度是（B x n x d)。该函数的两个输入必须是三维矩阵且第一维相同(表示Batch维度)，#不支持broadcast操作
```

###  "混合"矩阵乘法 torch.matmul() **支持broadcast操作**

```python
torch.matmul(input, other, out=None) # 支持broadcast操作

特别，针对多维数据matmul()乘法，我们可以认为该 matmul()乘法使用 <使用两个参数的后两个维度来计算>，其他的维度都可以认为是batch维度。
假设两个输入的维度分别是input(1000 x 500×99 x 11), other(500 × 11 x 99)，那么我们可以认为torch.matmul(input, other)乘法首先是进行后两位矩阵乘法得到
(99 x 11)× (11 × 99) →(99 x 99)，然后分析两个参数的batch size分别是(1000 x 500)和500，可以广播成为（1000 x 500)，因此最终输出的维度是( 1000 x 500 x 99 x 99)
```

## 24.masked_fill_(mask, value)掩码操作，就地操作

用value填充tensor中与mask中**值为1位置相对应的元素**。mask的形状必须与要填充的tensor形状一致

```python
a = torch.randn(5,6)

x = [5,4,3,2,1]
mask = torch.zeros(5,6,dtype=torch.float)
for e_id, src_len in enumerate(x):
    mask[e_id, src_len:] = 1
mask = mask.to(device = 'cpu')
print(mask)
a.data.masked_fill_(mask.byte(),-float('inf'))
print(a)

# ------------输出----------------
tensor([[0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 1., 1., 1.],
        [0., 0., 1., 1., 1., 1.],
        [0., 1., 1., 1., 1., 1.]])
tensor([[-0.1053, -0.0352,  1.4759,  0.8849, -0.7233,    -inf],
        [-0.0529,  0.6663, -0.1082, -0.7243,    -inf,    -inf],
        [-0.0364, -1.0657,  0.8359,    -inf,    -inf,    -inf],
        [ 1.4160,  1.1594,    -inf,    -inf,    -inf,    -inf],
        [ 0.4163,    -inf,    -inf,    -inf,    -inf,    -inf]])
```

## 25.numpy np.triu() 取得上三角矩阵

`np.triu(a, k)`是取矩阵a的`上三角数据`，但这个三角的斜线位置由k的值确定

```python
a = [
    [ 1, 2, 3, 4],
    [ 5, 6, 7, 8],
    [ 9,10,11,12],
    [13,14,15,16]
]

当np.triu(a, k = 0)时，得到<自主对角线开始的上三角数据>，即
a = [
    [ 1, 2, 3, 4],
    [ 0, 6, 7, 8],
    [ 0, 0,11,12],
    [ 0, 0, 0,16]
]

当np.triu(a, k = 1)时，得到主对角线<向上平移一个距离的对角线>，也叫右上对角线及其以上的数据，即
a = [
    [ 0, 2, 3, 4],
    [ 0, 0, 7, 8],
    [ 0, 0, 0,12],
    [ 0, 0, 0, 0]
]

当np.triu(a, k = -1)时，得到主对角线向下平移一个距离的对角线，也叫左下对角线及其以上的数据，即
a = [
    [ 1, 2, 3, 4],
    [ 5, 6, 7, 8],
    [ 0,10,11,12],
    [ 0, 0,15,16]
]
```

`np.tirl()`是取矩阵`下三角数据`，k的取值含义同上，只是得到的是自对角线以下的数据

## 26.Pytorch中model.train()和model.eval()区别

> - `如果模型中有BN层(Batch Normalization)和Dropout`，需要在**训练时添加model.train()**，在**测试时添加model.eval()**
>   - model.train()是保证BN层每一层批数据的均值和方差，
>   - 而model.eval()是保证BN用全部数据的均值和方差
> - `对于Dropout`，**model.train()是随机取一部分网络连接来训练更新参数**，而**model.eval()是利用所有网络连接**

## 27.Pytorch中torch.chunk()方法对张量分块

**注意是等分，分块的数目要和对应维度的值一样**

```python
torch.chunk(tensor, chunks, dim=0) → List of Tensors
# 返回值是一个list的tensor
'''
tensor (Tensor) – 要被分割的tensor
chunks (int) – number of chunks to return（要分割成的块数）
dim (int) – dimension along which to split the tensor（在哪个维度上进行操作）
'''
```

<img src="https://gitee.com/KangPeiLun/images/raw/master/images/0f4f64ca95cc4d1ca1ed0bfc83603913.png" alt="img" style="zoom:67%;" />

<img src="https://gitee.com/KangPeiLun/images/raw/master/images/498a511106a545c4a424e33e90914544.png" alt="img" style="zoom:67%;" />



## 28.python中`[i:j]`和`[i:j:k]`的区别

```python
a=[1,2,3,4,5]
print(a)	# [1,2,3,4,5]

print(a[-1])  # 输出最后一个元素 [5]

print(a[:-1])  # 输出除最后一个元素的所有元素 [1,2,3,4]

print(a[::-1])  # 倒序输出所有元素	[5,4,3,2,1]

print(a[2::-1])  # 倒序输出第3个元素之后的元素	[3,2,1]

print(a[1::2])  # 从第2个元素起，步长为2取元素	[2,4]
```

### b=a[i:j]

> b=a[i:j]表示复制a[i]到a[j-1]，赋值给b.
>
> 当i缺省时，默认为0，即 a[:3]相当于 a[0:3]
>
> 当j缺省时，默认为len(alist), 即a[1:]相当于a[1:len(alist)]
>
> 当i，j都缺省时，a[:] 就相当于完整复制一份a

### b=a[i:j:k]

> 在上面的b=a[i:j]基础上`加了个步长k`。k缺省值为1.
>
> 当k<0时，i缺省时，默认为-1. j缺省时，默认为-len(a)-1	`k<0时，可以理解为倒序`
>
> 所以a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍，即倒序。

对于其他参数σ和λi以及超像素聚类方法带宽的主要参数，不规则约束中的替代参数σ可以从{1,10,100,200}调整。 带宽用于控制从集合{10−3, 10−2,..., 10, 102}中选择的mean-shift模型的质心圆半径，最终的选择是基于性能 聚类结果。

## 29.Pytorch中设置随机种子，以使得结果可以被复现

**把下面的函数当作一种标准的写法**

```python
import random
import numpy as np
import torch

def same_seeds(seed):
    '''
    # TODO: 学习这种写法
    将随机种子设置为某个值 以实现可重复性
    :param seed:
    :return:
    '''
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)     # 为CPU设置种子用于生成随机数，以使得结果是确定的
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)    # 为当前GPU设置随机种子，以使得结果是确定的
        torch.cuda.manual_seed_all(seed)    # 为所有的 GPU 设置种子用于生成随机数，以使得结果是确定的
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
```

###  torch.backends.cudnn.benchmark = True

> ```python
> torch.backends.cudnn.benchmark = True 
> 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
> 一般来讲，应该遵循以下准则:
>     如果网络的输入数据维度或类型上变化不大，网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小尺寸，输入的通道）是不变的，
>     设置 torch.backends.cudnn.benchmark = true 可以增加运行效率
> 反之:
>     如果网络的输入数据在每次 iteration 都变化的话，（例如，卷积层的设置一直变化、某些层仅在满足某些条件时才被“激活”，或者循环中的层可以重复不同的次数），
>     会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会耗费更多的时间，降低运行效率
> ```

### torch.backends.cudnn.deterministic = True

> ```python
> benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异。
> 如果想要避免这种结果波动，设置：torch.backends.cudnn.deterministic = True保证实验的可重复性
> ```

## 30.nn.Conv1d和nn.Conv2d卷积、nn.ConvTranspose2d反卷积、上下采样函数interpolate()

### nn.Conv1d 一维的卷积能处理多维数据

> ```python
> nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
> 
> # -----------------------------------
> # 例子
> import torch
> import torch.nn as nn
> import torch.nn.functional as F
> 
> x = torch.randn(10, 16, 30, 32, 34)
> # batch, channel , height , width
> print(x.shape)
> class Net_1D(nn.Module):
>     def __init__(self):
>         super(Net_1D, self).__init__()
>         self.layers = nn.Sequential(
>             nn.Conv1d(in_channels=16, out_channels=16, kernel_size=(3, 2, 2), stride=(2, 2, 1), padding=[2,2,2]),
>             nn.ReLU()
>         )
>     def forward(self, x):
>         output = self.layers(x)
>         log_probs = F.log_softmax(output, dim=1)
>         return  log_probs
> 
> n = Net_1D()  # in_channel,out_channel,kennel,
> print(n)
> y = n(x)
> print(y.shape)
> 
> # -----------------------------------
> # 输出
> torch.Size([10, 16, 30, 32, 34])
> Net_1D(
>   (layers): Sequential(
>     (0): Conv1d(16, 16, kernel_size=(3, 2, 2), stride=(2, 2, 1), padding=[2, 2, 2])
>     (1): ReLU()
>   )
> )
> torch.Size([10, 16, 16, 18, 37])
> ```
>
> #### 参数：
>
>   `in_channel`:　输入数据的通道数，例RGB图片通道数为3；
>
>   `out_channel`: 输出数据的通道数，这个根据模型调整；
>
>   `kennel_size`: 卷积核大小，可以是int，或tuple；kennel_size=2,意味着卷积大小2， kennel_size=（2,3），意味着卷积在第一维度大小为2，在第二维度大小为3；
>
>   `stride`：步长，默认为1，与kennel_size类似，stride=2,意味在所有维度步长为2， stride=（2,3），意味着在第一维度步长为2，意味着在第二维度步长为3；
>
>   `padding`：　零填充
>
> ​	 `group`: group用于设置卷积分组数，如果`大于1即为分组卷积`
>
> #### 卷积计算
>
> d = (d - kennel_size + 2 * padding) / stride + 1
>
> x = ([10,16,30,32,34]),其中第一维度：30，第一维度,第二维度：32,第三维度：34，对于卷积核长分别是；
>
> 对于步长分别是第一维度：2,第二维度：,2,第三维度：1；对于padding分别是：第一维度：2,第二维度：,2,第三维度：2；
>
> d1 = (30 - 3 + 22)/ 2 +1 = 31/2 +1 = 15+1 =16
>
> d2 = (32 - 2 + 22)/ 2 +1 = 34/2 +1 = 17+1 =18
>
> d3 = (34 - 2 + 2*2)/ 1 +1 = 36/1 +1 = 36+1 =37
>
> batch = 10, out_channel = 16
>
> 故：y = [10, 16, 16, 18, 37]



### nn.Conv2d 二维卷积可以处理二维数据

> ```python
> nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
> 
> # -----------------------------------
> # 例子
> import torch
> import torch.nn as nn
> import torch.nn.functional as F
> 
> x = torch.randn(10, 16, 30, 32, 34)
> # batch, channel , height , width
> print(x.shape)
> class Net_1D(nn.Module):
>     def __init__(self):
>         super(Net_1D, self).__init__()
>         self.layers = nn.Sequential(
>             nn.Conv1d(in_channels=16, out_channels=16, kernel_size=(3, 2, 2), stride=(2, 2, 1), padding=[2,2,2]),
>             nn.ReLU()
>         )
>     def forward(self, x):
>         output = self.layers(x)
>         log_probs = F.log_softmax(output, dim=1)
>         return  log_probs
> 
> n = Net_1D()  # in_channel,out_channel,kennel,
> print(n)
> y = n(x)
> print(y.shape)
> 
> # -----------------------------------
> # 输出
> torch.Size([10, 16, 30, 32, 34])
> Net_1D(
>   (layers): Sequential(
>     (0): Conv1d(16, 16, kernel_size=(3, 2, 2), stride=(2, 2, 1), padding=[2, 2, 2])
>     (1): ReLU()
>   )
> )
> torch.Size([10, 16, 16, 18, 37])
> ```
>
> #### 参数：
>
>   `in_channel`:　输入数据的通道数，例RGB图片通道数为3；
>
>   `out_channel`: 输出数据的通道数，这个根据模型调整；
>
>   `kennel_size`: 卷积核大小，可以是int，或tuple；kennel_size=2,意味着卷积大小2， kennel_size=（2,3），意味着卷积在第一维度大小为2，在第二维度大小为3；
>
>   `stride`：步长，默认为1，与kennel_size类似，stride=2,意味在所有维度步长为2， stride=（2,3），意味着在第一维度步长为2，意味着在第二维度步长为3；
>
>   `padding`：　零填充
>
> #### 卷积计算
>
> d = (d - kennel_size + 2 * padding) / stride + 1
>
> x = ([10,16,30,32,34]),其中第一维度：30，第一维度,第二维度：32,第三维度：34，对于卷积核长分别是；
>
> 对于步长分别是第一维度：2,第二维度：,2,第三维度：1；对于padding分别是：第一维度：2,第二维度：,2,第三维度：2；
>
> d1 = (30 - 3 + 22)/ 2 +1 = 31/2 +1 = 15+1 =16
>
> d2 = (32 - 2 + 22)/ 2 +1 = 34/2 +1 = 17+1 =18
>
> d3 = (34 - 2 + 2*2)/ 1 +1 = 36/1 +1 = 36+1 =37
>
> batch = 10, out_channel = 16
>
> 故：y = [10, 16, 16, 18, 37]

### nn.ConvTranspose2d反卷积——`上采样，将特征图尺寸变大`

> `反卷积是卷积的逆过程`，又称作转置卷积。最大的区别在于`反卷积过程是有参数要进行学习的`（类似卷积过程），理论是反卷积可以实现UnPooling和unSampling，只要卷积核的参数设置的合理。

```python
nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
```

- `padding`(int or tuple, optional) - 输入的每一条边补充0的层数，高宽都增加2*padding

- `output_padding`(int or tuple, optional) - 输出边补充0的层数，高宽都增加padding

- 输出尺寸计算：
  **output = (input-1)stride+outputpadding -2padding+kernelsize**

```python
'''反卷积操作实例'''
import torch
from torch import nn
from torch.nn import functional as F


    def __init__(self):
        ......  # 省略部分代码
        self.convTrans6 = nn.Sequential(
            # ConvTranspose2d 反卷积操作
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(2,2),
                               padding=(0,0)),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(3,3), stride=(2,2),
                               padding=(0,0)),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(3,3), stride=(2,2),
                               padding=(0,0)),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()  # y = (y1, y2, y3) \in [0 ,1]^3
        )
	
    def forward(self, x)
        ......
        x = self.relu(self.fc5(x)).view(-1, 64, 4, 4)  # 输入反卷积前，需要先将 x 的形状变为 3 维的
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)  # [batch_size, 3, 39, 39]
        # interpolate 可以理解为 因为反卷积不能保证最后的尺寸是合适的，所以用interpolate进行最后的调整
        # 反卷积虽然可以增大特征图的尺寸，但其核心目的是调整通道数
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
```

### Pytorch上下采样函数**interpolate()注意维度**——**和nn.ConvTranspose2d反卷积搭配使用**

> interpolate 可以理解为 因为`反卷积不能保证最后的尺寸是合适的`，所以`用interpolate进行最后的调整`
>
> `反卷积虽然可以增大特征图的尺寸`，但其`核心`目的是`调整通道数`

#### torch.nn.functional.interpolate实现插值和上采样

> 上采样，在深度学习框架中，可以简单的理解为任何可以让你的图像变成更高分辨率的技术。 最简单的方式是重采样和插值：将输入图片input image进行rescale到一个想要的尺寸，而且计算每个点的像素点，使用如双线性插值bilinear等插值方法对其余点进行插值。
>
> Unpooling是在CNN中常用的来表示max pooling的逆操作。因为max pooling不可逆，因此使用近似的方式来反转得到max pooling操作之前的原始情况

```python
torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
# 注意：如果input的shape是4维的，则 size参数必须为2维的 如input shape:(1, 1, 256, 256)  size=(256, 256)
#	   如果input的shape是3维的，则 size参数必须为1维的 如input shape:(1, 1, 256, 256)  size=(256,)
#	   否则会报错
```

- `input `(Tensor) – `输入张量`

- `size` (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]) – `输出大小`.

- `scale_factor` (float or Tuple[float]) – `指定输出为输入的多少倍数`。如果输入为tuple，其也要制定为tuple类型

- `mode` (str) – 可使用的上采样算法，有`’nearest’, ‘linear’, ‘bilinear’, ‘bicubic’ , ‘trilinear’和’area’`. 默认使用’nearest’

  > 注：使用mode='bicubic’时，可能会导致overshoot问题，即它可以为图像生成负值或大于255的值。如果你想在显示图像时减少overshoot问题，可以显式地调用result.clamp(min=0,max=255)。

- `align_corners` (bool, optional) – 几何上，`我们认为输入和输出的像素是正方形`，而不是点。如果设置为`True，则输入和输出张量由其角像素的中心点对齐，从而保留角像素处的值`。如果设置为False，则输入和输出张量由它们的角像素的角点对齐，插值使用边界外值的边值填充;当scale_factor保持不变时，使该操作独立于输入大小。仅当使用的算法为’linear’, ‘bilinear’, 'bilinear’or 'trilinear’时可以使用。默认设置为False

  > 如果 align_corners=True，则对齐 input 和 output 的角点像素(corner pixels)，保持在角点像素的值. 只会对 mode=linear, bilinear 和 trilinear 有作用. 默认是 False.

> 根据给定的`size或scale_factor`参数来对输入进行下/上采样，一般`size和scale_factor只使用其中一个参数，另一个不设置`，如下：
>
> 使用的插值算法取决于参数mode的设置
>
> ```python
> x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
> ```



## 31.Tensor中常见基本操作

详见：[Tensor及其基本操作](https://zhuanlan.zhihu.com/p/36233589)

<img src="https://gitee.com/KangPeiLun/images/raw/master/images/v2-ef4d55e72864bd645ec7112c642b68cf_r.jpg" alt="preview" style="zoom:67%;" />

### 提醒

```python
# 只要是这样连续点的写法，计算顺序都是从左到右进行的
std = logvar.mul(0.5).exp_()
# 这个式子等价于下面的公式
```

$$ {std = logvar.mul(0.5).exp_()}
std = e^{logvar*0.5}
$$

#### torch.bmm(a,b) 两个tensor矩阵乘法——两个tensor的维度必须为3

计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,h),注意两个tensor的维度必须为3

## 32.VAE变分自编码器标准写法

<img src="https://gitee.com/KangPeiLun/images/raw/master/images/1183530-20170928111601762-929103944.png" alt="img" style="zoom:67%;" />

**以后都按照这个标准流程进行书写**

```python
'''
	1.将encode的输出拆分成shape相同的两个 mu和logvar
		可以把encode的输出直接 运算两次得到，如下面的enc_out_1(mu), enc_out_2(logvar)  这样的两个参数就可以通过网络学习到
	2.将 mu 和 logvar 丢进函数 reparametrize() 计算 h
	3.将 h 传入 decoder 得到网络的输出
	4.网络需要返回 decoder输出、mu、logvar
	5.使用 decoder输出和原数据 计算 MSE Loss；使用 mu和logvar 计算kld_loss即为KL-divergence（KL散度），用来衡量潜在变量的分布和单位高斯分布的差异
		
'''

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # 注意生成mu和logvar的特征图通道数是翻倍的
        self.enc_out_1 = nn.Sequential(  # mu
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.enc_out_2 = nn.Sequential(  # logvar
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        h1 = self.encoder(x)
        return self.enc_out_1(h1), self.enc_out_2(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()   # 初始化一个shape为 std.size() 的网络参数，默认初始化值取自标准正态分布(mean=0, std=1)
        # TODO: Variable的作用
        '''
        Variable就是 变量 的意思。实质上也就是可以变化的量，区别于int变量，它是一种可以变化的变量，这正好就符合了反向传播，参数更新的属性。
        在pytorch中的Variable就是一个存放会变化值的地理位置，里面的值会不停发生片花，就像一个装鸡蛋的篮子，鸡蛋数会不断发生变化。那谁是里面的鸡蛋呢，自然就是pytorch中的tensor了。
        （也就是说，pytorch都是有tensor计算的，而tensor里面的参数都是Variable的形式）。如果用Variable计算的话，那返回的也是一个同类型的Variable
        '''
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images 网络预测的图片结果
    x: origin images 原图片
    mu: latent mean
    logvar: latent log variance 潜在对数方差
    criterion: 用于计算 预测结果 和 原始数据 的loss 
    """
    mse = criterion(recon_x, x)  # mse loss
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)  # kld_loss即为KL-divergence（KL散度），用来衡量潜在变量的分布和单位高斯分布的差异
    return mse + KLD


#---------------------------------------------
# 使用方法
output = model(img)		# 模型的返回值用一个变量接收，则为一个元组 (pred_output, mu, logvar)
loss = loss_vae(output[0], img, output[1], output[2], criterion)   
```

## 33.Python装饰器abstractmethod、property、classmethod、staticmethod及自定义装饰器

详见：[Python装饰器abstractmethod、property、classmethod、staticmethod及自定义装饰器](https://blog.csdn.net/weixin_41624982/article/details/87650504)

> - @`abstractmethod`：抽象方法，`含abstractmethod方法的类不能实例化`，`继承`了含`abstractmethod方法的子类必须复写所有abstractmethod装饰的方法`，**未被装饰的可以不重写**
> - @ `property`：方法伪装属性，方法返回值及属性值，被装饰方法不能有参数，必须实例化后调用，类不能调用
> - @ `classmethod`：类方法，可以通过实例对象和类对象调用，被该函数修饰的方法第一个参数代表类本身常用cls，被修饰函数内可调用类属性，不能调用实例属性
> - @`staticmethod`：静态方法，可以通过实例对象和类对象调用，被装饰函数可无参数，被装饰函数内部通过类名.属性引用类属性或类方法，不能引用实例属性

### @abstractmethod 抽象方法声明

用于程序接口的控制，正如上面的特性，含有@abstractmethod修饰的父类不能实例化，但是继承的子类必须实现@abstractmethod装饰的方法

```python
 from abc import ABC, abstractmethod
 
class **A**(**ABC**):
    @abstractmethod
    def **test**(self):
    pass
 
class **B**(**A**):
    def **test_1**(self):
    print("未覆盖父类abstractmethod")
 
class **C**(**A**):
    def **test**(self):
    print("覆盖父类abstractmethod")
 
if __name__ == '__main__':
    a = A()
    b = B()
    c = C()
    前两个分别报错如下：
    a = A()
    TypeError: Can't instantiate abstract class A with abstract methods test

    b = B()
    TypeError: Can't instantiate abstract class **B** **with** **abstract** **methods** **test**
    第三个实例化是正确的
```

### @ property 方法伪装属性

将一个方法伪装成属性，被修饰的特性方法，内部可以实现处理逻辑，但对外提供统一的调用方式，实现一个实例属性的get，set，delete三种方法的内部逻辑，具体含义看示例code。

```python
class **Data**:
    
    def **__init__**(self):
    self.number = 123
 
	@property
    def **operation**(self):
    return self.number

    @operation.setter
    def **operation**(self, number):
    self.number = number

    @operation.deleter
    def **operation**(self):
    del self.number
```

### @ classmethod类方法  @staticmethod静态方法

类方法classmethod和静态方法staticmethod是为类操作准备，是将类的实例化和其方法解耦，`可以在不实例化的前提下调用某些类方法`。两者的区别可以这么理解：`类方法是将类本身作为操作对象`，而`静态方法是独立于类的一个单独函数，只是寄存在一个类名下`。类方法可以用过类属性的一些初始化操作。

```python
# -*- coding:utf-8 -*-
 
class **Test**:
    num = "aaaa"

    def **__init__**(self):
    self.number = 123

    @classmethod
    def **a**(cls, n):
    cls.num = n
    print(cls.num)

    @classmethod
    def **b**(cls, n):
    cls.a(n)

    @classmethod
    def **c**(cls, n):
    cls.number = n

    @staticmethod
    def **d**(n):
        
Test.b(n)
```

## 34.Pytorch Lightning 轻量级 PyTorch 包装器

**这种方法虽然方便，但现在先不作为我的主流代码构建方式**

官方文档：[PyTorch Lightning 1.5.9 documentation](https://pytorch-lightning.readthedocs.io/en/stable/)

基本使用：[PyTorchLightning基本使用](https://github.com/PyTorchLightning/pytorch-lightning/tree/deb1581e26b7547baf876b7a94361e60bb200d32)

详见：[PyTorchLightning 轻量级Pytorch包装器)](https://github.com/PytorchLightning/pytorch-lightning#how-do-i-do-use-it) `相当于在Pytorch的基础上有进行了一次封装`，使得代码构建起来更轻松

### 快速使用

#### 1.导入这些包

```python
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
```

#### 2.定义一个 LightningModule (nn.Module subclass)

LightningModule 定义了一个完整的系统

`Training_step 定义了训练循环。 Forward 定义 LightningModule 在推理/预测期间的行为方式`

```python
class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```

#### 3.训练

```python
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])

autoencoder = LitAutoEncoder()
trainer = pl.Trainer()
trainer.fit(autoencoder, DataLoader(train), DataLoader(val))
```

## 35.np.argwhere()用法

```python
np.argwhere(a) 

返回非 0 的数组元组的索引，其中 a 是要索引数组的条件
```

![img](https://gitee.com/KangPeiLun/images/raw/master/images/20180323205039740)

> 比如此例子中，将x数组中所有大于1的元素 对应在数组中的坐标返回
>
> 满足 x>1 的一共有4个元素，分别是2，3，4，5；对应的坐标分别是[0,2] [1,0] [1,1] [1,2]

## 36.混淆矩阵及confusion_matrix()函数的使用

混淆矩阵是机器学习中总结分类模型预测结果的情形分析表，以矩阵形式将数据集中的记录按照真实的类别与分类模型作出的分类判断两个标准进行汇总。`这个名字来源于它可以非常容易的表明多个类别是否有混淆`（`也就是一个class被预测成另一个class`）

![img](https://gitee.com/KangPeiLun/images/raw/master/images/20170814211735042)

其中`灰色部分是真实分类和预测分类结果相一致的`，**蓝色部分是真实分类和预测分类不一致的，即分类错误的**。

### confusion_matrix函数的使用

```python
sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
```

> - `y_true`: 是样本真实分类结果
> - `y_pred`: 是样本预测分类结果 
> - `labels`: 是所给出的类别，通过这个可对类别进行选择 
> - `sample_weight`: 样本权重

```python
# 例子
from sklearn.metrics import confusion_matrix
 
y_true=[2,1,0,1,2,0]
y_pred=[2,0,0,1,2,1]
 
C=confusion_matrix(y_true, y_pred)
```

> PS：**横轴表示真实分类，纵轴表示预测分类**
>
> 运行结果：
>
> ![img](https://gitee.com/KangPeiLun/images/raw/master/images/20170814215527266)
>
> 下图是标注类别以后，更加好理解：
>
> ![img](https://gitee.com/KangPeiLun/images/raw/master/images/20170814220712046)

## 37.OA、AA_mean、Kappa、AA四个指标标准计算函数

**以后计算这四个指标时，可以直接使用该代码**

```python
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA

#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
```

## 38.&和&&的区别，其他同理

> - 使用`&`时，`当前面的表达式为假`的时候，程序`依旧会继续执行后面的表达式`，然后再得出FALSE的结果
> - 当使用`&&（短路与）`时，则相反，当`前面的表达式结果为假`时则`不会再执行后面的表达式`，直接得出FALSE的结果

## 39.np.c_ 和 np.r_ 的用法解析

> `np.r_`是按列连接两个矩阵，就是把两矩阵上下相加，`要求列数相等`。 **结果是增加了行数**
>
> `np.c_`是按行连接两个矩阵，就是把两矩阵左右相加，`要求行数相等`。 **结果是增加了列数**

### np.r_用法示例——`结果是增加了行数`

```python
a = np.array([[1, 2, 3],[7,8,9]])
b=np.array([[4,5,6],[1,2,3]])
 
d= np.array([7,8,9])
e=np.array([1, 2, 3])
 
g=np.r_[a,b]
 
g
Out[14]: 
array([[1, 2, 3],
       [7, 8, 9],
       [4, 5, 6],
       [1, 2, 3]])  # 结果是增加了行数
 
h=np.r_[d,e]
 
h
Out[16]: array([7, 8, 9, 1, 2, 3])
```

### np.c_用法示例——`结果是增加了列数`

```python
a = np.array([[1, 2, 3],[7,8,9]])
 
b=np.array([[4,5,6],[1,2,3]])
 
a
Out[4]: 
array([[1, 2, 3],
       [7, 8, 9]])
 
b
Out[5]: 
array([[4, 5, 6],
       [1, 2, 3]])
 
c=np.c_[a,b]
 
c
Out[7]: 
array([[1, 2, 3, 4, 5, 6],
       [7, 8, 9, 1, 2, 3]])  # 结果是增加了列数
 
 
 
d= np.array([7,8,9])
 
e=np.array([1, 2, 3])
 
f=np.c_[d,e]
 
f
Out[12]: 
array([[7, 1],
       [8, 2],
       [9, 3]])
```

## 40.nn.Identity()用法——用于模型结构占位

**Identity模块不改变输入，直接return input**

> 一种编码技巧吧，比如我们要加深网络，有些层是不改变输入数据的维度的，
>
> 在增减网络的过程中我们就可以`用identity占个位置`，`这样网络整体层数永远不变，看起来可能舒服一些`

```python
self.fc1 = nn.Identity()  # 保持结构完整，好看，用于占位，并不参与实际前向传播

(layer4): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    (fc): Linear(in_features=2048, out_features=1024, bias=True)
  )
  (fc1): Identity()		# 可以看到Identity()什么都没有改变，仅仅是占了个位子
  (bn1): BatchNorm1d(1024, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=1024, out_features=768, bias=True)
  (bn2): BatchNorm1d(768, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
```

## 41.datetime()自定义时间格式

> 两个时间相关的库
>
> ```python
> import time
> import datetime
> ```
>
> 获取当前日期和时间：
>
> ```python
> now_tm = datetime.datetime.now()
> print(now_tm)
> ```
>
> 可以格式化想要的日期格式：
>
> ```python
> now_tm = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
> print(now_tm)  # 输出秒级的时间-年月日时分秒
> ```
>
> **格式参考**
>
> - %a 星期几的简写
> - `%A 星期几的全称`
> - %b 月分的简写
> - `%B 月份的全称`
> - %c 标准的日期的时间串
> - %C 年份的后两位数字
> - `%d 十进制表示的每月的第几天`
> - %e 在两字符域中，十进制表示的每月的第几天
> - %F 年-月-日
> - %g 年份的后两位数字，使用基于周的年
> - %G 年分，使用基于周的年
> - %h 简写的月份名
> - `%H 24小时制的小时`
> - %I 12小时制的小时
> - %j 十进制表示的每年的第几天
> - `%m 十进制表示的月份`
> - `%M 十时制表示的分钟数`
> - %n 新行符
> - %p 本地的AM或PM的等价显示
> - %r 12小时的时间
> - %R 显示小时和分钟：hh:mm
> - `%S 十进制的秒数`

## 42.python对图像进行padding，扩展图像

**学习标准写法**

```python
def mirror_hsi(height,width,band,input_normalize,patch=5):
    ''' 为了方便描述，将HSI影像表述为 图像
    因为作者是将HSI影像中相邻的像素合并作为输入，因此需要对原图像进行padding，进而扩展图像，否则有些像素点可能找不到相邻像素点
    拓展的准则是 对原图像 进行上下左右填充，填充的内容为相邻边界处的像素值
    比如：
        原图像为：[1, 2], 那么向右padding=2时，则产生的镜像为：[1, 2, 2, 2]， 全都用 2 向右填充
    :param height:
    :param width:
    :param band:
    :param input_normalize: 表示经过标准化后的 原图像
    :param patch: 根据 相邻多少个像素 合为一起 来计算 padding多少圈  padding=patch//2
    :return:
    '''
    # 假设patch=2，则padding=1
    padding=patch//2
    # 事先创建好padding后镜像的大小，用于占位，此时mirror_hsi的shape为[147, 147, 200]
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize  # 把原图像的内容对应放到镜像中去，对应下图蓝色部分
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]  # 对应下图黄色部分
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]  # 对应下图橘色部分
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]  # 对应下图红色部分
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]  # 对应下图绿色部分
        
    return mirror_hsi
```

![image-20220213222957432](https://gitee.com/KangPeiLun/images/raw/master/images/image-20220213222957432.png)

# 43.pytorch使用torchvision库加载官方定义好的模型

```python
import torch
from torchvision import models  # models 类中定义了各种官方定义好的模型，可以直接加载使用

def initial_backbone(model_name='resnet18', use_pretrained=True, num_classes=1000):
    '''
    初始化 backbone, 指的是各个版本的resnet
    :param model_name: 初始化模型的名字 choose in ['resnet18', 'resnet34', 'resnet50']
    :param use_pretrained: 是否使用预训练权重
    :return:
    '''
    if model_name == 'resnet18':
        backbone = models.resnet18(pretrained=use_pretrained)
        num_ftrs = backbone.fc.in_features   # 获取模型fc层中，in_features 的数量   <通过 . 的方式可以获取实例化后的模型的属性信息>
        # print(num_ftrs)
        # model_data.fc 表示重写 resnet152模型的fc层
        backbone.fc = nn.Sequential(					# <通过 再次赋值 的方式可以 更改 实例化后的 模型的 layer, 操作跟正常定义模型一样>
            nn.Linear(num_ftrs, num_classes),
            nn.LogSoftmax(dim=1),    # 将输出变为一维向量
        )

    elif model_name == 'resnet34':
        backbone = models.resnet34(pretrained=use_pretrained)

    elif model_name == 'resnet50':
        backbone = models.resnet50(pretrained=use_pretrained)

    else:
        raise ValueError(f'Your selected model({model_name}) not in "resnet18,resnet34,resnet50"')

    return backbone
```

### 导入预训练模型的部分结构

有时候，我们想利用一些经典模型的架构作为特征提取器，仅想导入部分结构，而不是整个模型。

以`densenet121为例`，例如仅`利用denseblock1及之前的结构+Transition`

首先检查densenet121的结构， 调用部分结构即可，例如：

**单独把你需要的那一层通过索引的方式取出来即可**

```python
class mydensenet121(nn.Module): # [N, 3, 224, 224] --> [N, 128, 56, 56]
    def __init__(self):
        super(mydensenet121, self).__init__()
        densenet_pretrained = torch_models.densenet121(pretrained=True)
        self.features = densenet_pretrained.features[:5] # 预处理层+denseblock1(denselayer1:6)
        self.bn_relu_1x1conv = densenet_pretrained.features[5][:3]


    def forward(self, x):
        out = self.features(x)
        out = self.bn_relu_1x1conv(out)
        return out
```

# 44.torch.pairwise_distance计算特征图之间对应元素 像素级 距离

```python
import torch.nn.functional as F
dist = F.pairwise_distance(x_A, x_B, keepdim=True)  # 特征距离

计算特征图之间的对应像素级的距离，输入x1维度为[N,C,H,W]，输入x2的维度为[M,C,H,W]
其中要求N==M or N==1 or M==1
```

# 45.快速将特征图变为二值图像(技巧)

```python
a = np.array([3, 123, 0, -1])
a = a>1
# 输出
array([ True,  True, False, False])   # 可以通过这种方法把特征图转换为 二值图像，大于1的都变为True，小于1的变为False
```

# 46.错误解决view size is not compatible with input tensor's size and stride

这是因为`view()需要Tensor中的元素地址是连续的`，但可能出现Tensor不连续的情况，所以先`用 .contiguous() 将其在内存中变成连续分布`：

```python
out = out.contiguous().view(out.size()[0], -1)
```

# 47.pytorch按照某一特定维度进行shuffle

```python
# example1
t=torch.tensor([[1, 2, 3],[3, 4, 5]])
print(t)
idx = torch.randperm(t.shape[1])  # t.shape[1] 表示你要shuffle的那个维度
t = t[:, idx].view(t.size())  	 # 这两步是组合使用的
print(t)


# example2
# 为了防止挑不出足够的 像素点，用原图像上随机的number个点进行替代
idx = torch.randperm(A.shape[2])
shuffle_A = A[:, :, idx].view(A.shape)
shuffle_B = B[:, :, idx].view(B.shape)

for batch_idx in range(batch):
    temp_index = index[batch_idx, :, :]

    # 获取 no change 像素点所对应的索引
    loc = torch.where(temp_index != 1)[1]  # np.where会对每一维数据进行判断，而这里与需要取第三维即可
    if loc.shape[0] != number:
        # 为了防止挑不出足够的 像素点，用原图像上随机的number个点进行替代
        temp_A[batch_idx, :, :] = shuffle_A[batch_idx, :, :number]
        temp_B[batch_idx, :, :] = shuffle_B[batch_idx, :, :number]
```

# 48.pytorch冻结网络权重

```python
def set_parameter_requires_grad(model, feature_extracting, freeze_layer):
    '''
    冻结网络中不需要训练的层
    :param model: 实例化后的模型
    :param feature_extracting:  是否对模型权重进行梯度更新，True表示不更新部分层，False表示全都更新
    :param freeze_layer: 要冻结的层数，注意要小于传入的网络的总层数
    :return: 无返回值，直接对model对象本身进行了修改
    '''
    print('Params to learn:')
    if feature_extracting:
        total_layers = len([param for param in model.parameters()])  # 记录该网络共有多少层的参数可以学习
        assert (freeze_layer < total_layers), f'the total_layers is {total_layers}, but freeze_layer is {freeze_layer}'
        print('total layers:',total_layers)
        '''
              model_data.named_parameters() 会返回模型的对应层的 name 和 param
        '''
        for index, (name,param) in enumerate(model.named_parameters()):
            if index < freeze_layer:
                # 冻结前freeze_layer层网络
                param.requires_grad = False

            # if param.requires_grad == True:
            #     # 冻结前400层后, 剩下的层数需要进行训练
            #     params_to_update.append(param)
            # print(param.requires_grad, len(params_to_update))
```

# 49.pytorch实现注意力机制

各种注意力机制实现：[pytorch中加入注意力机制(CBAM)，以ResNet为例，解析到底要不要用ImageNet预训练？如何加预训练参数？](https://zhuanlan.zhihu.com/p/99261200?from=singlemessage)

### 空间注意力机制

```python
import torch
import torch.nn.functional as F
from torch import nn


class BAM(nn.Module):
    """ Basic self-attention module
    """
    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(BAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in // 8
        self.activation = activation
        self.ds = ds  # 下采样倍数
        self.pool = nn.AvgPool2d(self.ds)
        # print('ds: ',ds)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)  # q   q.shape==k.shape
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)    # k
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)       # v
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, input):
        """
            inputs : [b, 64, 64, 128]
                input feature maps(B, C, W, H)
            returns :
                out : self attention value + input feature
                attention: (B, N, N)  N is Width*Height
        """
        x = self.pool(input) # [b, 64, 64, 128]
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds) [b, 64*128, 64//8]
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)  [b, 64//8, 64*128]
        '''
        torch.bmm()
        计算两个tensor的矩阵乘法，torch.bmm(a,b)
        tensor a 的size为(b,h,w)  tensor b的size为(b,w,h)
        注意两个tensor的维度必须为3.
        '''
        energy = torch.bmm(proj_query, proj_key)  # energy活力 transpose check  q*k   [b, 64*128, 64*128]
        energy = (self.key_channel**-.5) * energy   # <=> (1/key_channel)^0.5

        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)  得到注意力分数  [b, 64*128, 64*128]

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N  [b, 64, 64*128]

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [b, 64, 64*128]
        out = out.view(m_batchsize, C, width, height)  # [b, 64, 64, 128]

        # self.ds 表示下采样的倍数，且这里的width，height均已表示下采样后的尺寸，此时跟input的shape就不一样了
        # 因此在对out进行上采样，才能保证out和input的尺寸一致
        out = F.interpolate(out, [width*self.ds,height*self.ds])  # [b, 64, 64, 128]
        out = out + input  # 这是一个残差结构，在注意力学的不好的情况下 也能保证原来的效果不会变差

        return out
```

### 通道注意力机制

```python


```

# 50.nn.AdaptiveAvgPool2d()与nn.AvgPool2d()两种池化的区别

### nn.AvgPool2d() 输出尺寸需要提前手动计算

![img](https://gitee.com/KangPeiLun/images/raw/master/images/20191013210641271.png)

![img](https://gitee.com/KangPeiLun/images/raw/master/images/20191013210741375.png)

一般我们使用它的时候，只需要关注 kernel_size 、stride 与 padding 三个参数就行了，最后输出的尺寸为：

![img](https://gitee.com/KangPeiLun/images/raw/master/images/20191013211110773.png)

其中，x 表示输入的维度大小， y 表示对应输出的维度大小



### nn.AdaptiveAvgPool2d() 参数即为输出尺寸

![img](https://gitee.com/KangPeiLun/images/raw/master/images/20191013211746903.png)

`只需要关注输出维度的大小 output_size `，具体的实现过程和参数选择自动帮你确定了

# 51.python文本字符串对齐

**这样在写文件时可以将内容关于某一个字符对齐**

```python
print("abc".center (30,'-')) 
print("abc".ljust (30)+'|') 
print("abc".rjust (30)) 
```

![image-20220314115650714](https://gitee.com/KangPeiLun/images/raw/master/images/image-20220314115650714.png)

可以实现下面的效果：

![image-20220314115813759](https://gitee.com/KangPeiLun/images/raw/master/images/image-20220314115813759.png)

# 52.pytorch自定义学习率 lr_scheduler.LambdaLR

```python
def lambda_rule(epoch):
    # 自定义学习率，前100学习率保持不变，后100学习率持续衰减到0
    lr_l = 1.0 - max(0, epoch + 1 - 100) / float(100 + 1)  # 注意这里是先计算 除法，再计算 减法
    return lr_l
self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
```

# 53.python random模块随机取list中的某个值

```python
import random
from random import randint
 
'''
random.randint()随机生一个整数int类型，可以指定这个整数的范围，同样有上限和下限值，python random.randint。
random.choice()可以从任何序列，比如list列表中，选取一个随机的元素返回，可以用于字符串、列表、元组等。
random.sample()可以从指定的序列中，随机的截取指定长度的片断，不作原地修改。
'''

list_one=["name","age",18]
choice_num=random.choice(list_one)
print(choice_num)
```

