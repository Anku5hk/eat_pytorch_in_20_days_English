# 5-4, TensorBoard visualization

In our alchemy process, if we can use rich images to show the structure of the model, the change of indicators, the distribution of parameters, the input form and other information, it will undoubtedly enhance our insight into the problem and increase the fun of alchemy.

TensorBoard is just such a magical alchemy visualization aid. It was originally the younger brother of TensorFlow, but it can also work well with Pytorch. Even using TensorBoard in Pytorch is easier and more natural than using TensorBoard in TensorFlow.

The approximate process of visualizing using TensorBoard in Pytorch is as follows:

First specify a directory in Pytorch to create a torch.utils.tensorboard.SummaryWriter log writer.

Then, according to the information that needs to be visualized, use the log writer to write the corresponding information log into the directory we specify.

Finally, you can pass in the log directory as a parameter to start TensorBoard, and then you can enjoy watching movies in TensorBoard.

We mainly introduce the method of using TensorBoard to visualize the following information in Pytorch.

* Visual model structure: writer.add_graph

* Visual indicator changes: writer.add_scalar

* Visualization parameter distribution: writer.add_histogram

* Visualize the original image: writer.add_image or writer.add_images

* Visual manual drawing: writer.add_figure


```python

```

### One, visual model structure

```python
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchkeras import Model,summary
```

```python
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)
        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5)
        self.dropout = nn.Dropout2d(p = 0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64,32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y
        
net = Net()
print(net)
```

```
Net(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))
  (dropout): Dropout2d(p=0.1, inplace=False)
  (adaptive_pool): AdaptiveMaxPool2d(output_size=(1, 1))
  (flatten): Flatten()
  (linear1): Linear(in_features=64, out_features=32, bias=True)
  (relu): ReLU()
  (linear2): Linear(in_features=32, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)
```

```python
summary(net,input_shape= (3,32,32))
```

```
-------------------------------------------------- --------------
        Layer (type) Output Shape Param #
================================================= ==============
            Conv2d-1 [-1, 32, 30, 30] 896
         MaxPool2d-2 [-1, 32, 15, 15] 0
            Conv2d-3 [-1, 64, 11, 11] 51,264
         MaxPool2d-4 [-1, 64, 5, 5] 0
         Dropout2d-5 [-1, 64, 5, 5] 0
 AdaptiveMaxPool2d-6 [-1, 64, 1, 1] 0
           Flatten-7 [-1, 64] 0
            Linear-8 [-1, 32] 2,080
              ReLU-9 [-1, 32] 0
           Linear-10 [-1, 1] 33
          Sigmoid-11 [-1, 1] 0
================================================= ==============
Total params: 54,273
Trainable params: 54,273
Non-trainable params: 0
-------------------------------------------------- --------------
Input size (MB): 0.011719
Forward/backward pass size (MB): 0.359634
Params size (MB): 0.207035
Estimated Total Size (MB): 0.578388
-------------------------------------------------- --------------
```

```python
writer = SummaryWriter('./data/tensorboard')
writer.add_graph(net,input_to_model = torch.rand(1,3,32,32))
writer.close()
```

```python
%load_ext tensorboard
#%tensorboard --logdir ./data/tensorboard
```

```python
from tensorboard import notebook
#View the launched tensorboard program
notebook.list()
```

```python
#Start the tensorboard program
notebook.start("--logdir ./data/tensorboard")
#Equivalent to executing tensorboard --logdir ./data/tensorboard on the command line
#Can be opened in the browser http://localhost:6006/ view
```

![](./data/5-4-graph结构.png)

```python

```

### Second, visual indicator changes


Sometimes in the training process, if we can dynamically view the change curve of loss and various metrics in real time, it will undoubtedly help us to understand the training situation of the model more intuitively.

Note that writer.add_scalar can only visualize changes in the value of a scalar. Therefore, it is generally used for visual analysis of changes in loss and metric.


```python
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter



# f(x) = a*x**2 + b*x + the minimum value of c
x = torch.tensor(0.0,requires_grad = True) # x needs to be differentiated
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)

optimizer = torch.optim.SGD(params=[x],lr = 0.01)


def f(x):
    result = a*torch.pow(x,2) + b*x + c
    return(result)

writer = SummaryWriter('./data/tensorboard')
for i in range(500):
    optimizer.zero_grad()
    y = f(x)
    y.backward()
    optimizer.step()
    writer.add_scalar("x",x.item(),i) #Record the value of x in the step i in the log
    writer.add_scalar("y",y.item(),i) #Record the value of y in the step i in the log

writer.close()
    
print("y=",f(x).data,";","x=",x.data)
```

```
y = tensor(0.); x = tensor(1.0000)
```

![](./data/5-4-指标变化.png)

```python

```

### Three, visual parameter distribution


If you need to visualize the changes of model parameters (generally non-scalar) during training, you can use writer.add_histogram.

It can observe the trend of the histogram of the tensor value distribution with the training steps.

```python
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


# Create a normal distribution tensor simulation parameter matrix
def norm(mean,std):
    t = std*torch.randn((100,20))+mean
    return t

writer = SummaryWriter('./data/tensorboard')
for step,mean in enumerate(range(-10,10,1)):
    w = norm(mean,1)
    writer.add_histogram("w",w, step)
    writer.flush()
writer.close()
    

```
![](./data/5-4-张量分布.png)

```python

```

### Fourth, visualize the original image


If we do image-related tasks, we can also visualize the original image in tensorboard.

If you only write a piece of image information, you can use writer.add_image.

If you want to write multiple image information, you can use writer.add_images.

You can also use torchvision.utils.make_grid to combine multiple pictures into one picture, and then write it with writer.add_image.

Note that what is passed in is the tensor data in Pytorch representing the image information.


```python
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets


transform_train = transforms.Compose(
    [transforms.ToTensor()])
transform_valid = transforms.Compose(
    [transforms.ToTensor()])

```

```python
ds_train = datasets.ImageFolder("./data/cifar2/train/",
            transform = transform_train,target_transform = lambda t:torch.tensor([t]).float())
ds_valid = datasets.ImageFolder("./data/cifar2/test/",
            transform = transform_train,target_transform = lambda t:torch.tensor([t]).float())

print(ds_train.class_to_idx)

dl_train = DataLoader(ds_train, batch_size = 50, shuffle = True, num_workers=3)
dl_valid = DataLoader(ds_valid, batch_size = 50, shuffle = True, num_workers=3)

dl_train_iter = iter(dl_train)
images, labels = dl_train_iter.next()

# View only one picture
writer = SummaryWriter('./data/tensorboard')
writer.add_image('images[0]', images[0])
writer.close()

# Mosaic multiple pictures into one picture, separated by a black grid
writer = SummaryWriter('./data/tensorboard')
# create grid of images
img_grid = torchvision.utils.make_grid(images)
writer.add_image('image_grid', img_grid)
writer.close()

# Write multiple pictures directly
writer = SummaryWriter('./data/tensorboard')
writer.add_images("images",images,global_step = 0)
writer.close()
```

```
{'0_airplane': 0, '1_automobile': 1}
```

```python

```

![](./data/5-4-原始图像可视化.png)

```python

```

### Five, visual manual drawing


If we display the results of matplotlib drawing in tensorboard, we can use add_figure.

Note that, unlike writer.add_image, writer.add_figure needs to pass in the figure object of matplotlib.


```python
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets


transform_train = transforms.Compose(
    [transforms.ToTensor()])
transform_valid = transforms.Compose(
    [transforms.ToTensor()])

ds_train = datasets.ImageFolder("./data/cifar2/train/",
            transform = transform_train,target_transform = lambda t:torch.tensor([t]).float())
ds_valid = datasets.ImageFolder("./data/cifar2/test/",
            transform = transform_train,target_transform = lambda t:torch.tensor([t]).float())

print(ds_train.class_to_idx)
```

```
{'0_airplane': 0, '1_automobile': 1}
```

```python
%matplotlib inline
%config InlineBackend.figure_format ='svg'
from matplotlib import pyplot as plt

figure = plt.figure(figsize=(8,8))
for i in range(9):
    img,label = ds_train[i]
    img = img.permute(1,2,0)
    ax=plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d"%label.item())
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
```

![](./data/5-4-九宫格.png)

```python
writer = SummaryWriter('./data/tensorboard')
writer.add_figure('figure',figure,global_step=0)
writer.close()                         
```

![](./data/5-4-可视化人工绘图.png)


**If this book is helpful to you and want to encourage the author, remember to add a star⭐️ to this project and share it with your friends 😊!**

If you need to further communicate with the author on the understanding of the content of this book, please leave a message under the public account "Algorithm Food House". The author has limited time and energy and will respond as appropriate.

You can also reply to keywords in the background of the official account: **Add group**, join the reader exchange group and discuss with you.


![算法美食屋logo.png](./data/算法美食屋二维码.jpg)


```python

```

```python

```
