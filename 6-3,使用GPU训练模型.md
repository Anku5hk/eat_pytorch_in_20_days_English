# 6-3, use GPU to train model


The training process of deep learning is often very time-consuming. It is commonplace to train a model for several hours, and it is common to train for a few days, sometimes even dozens of days.

The time-consuming training process mainly comes from two parts, one part comes from data preparation, and the other part comes from parameter iteration.

When the data preparation process is still the main bottleneck of model training time, we can use more processes to prepare the data.

When the parameter iteration process becomes the main bottleneck of training time, our usual method is to use GPU to accelerate.

<!-- #region -->
Using GPU to accelerate the model in Pytorch is very simple, just move the model and data to the GPU. The core code has only the following lines.

```python
# Define model
...

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device) # Move model to cuda

# Training model
...

features = features.to(device) # Move data to cuda
labels = labels.to(device) # or labels = labels.cuda() if torch.cuda.is_available() else labels
...
```

If you want to use multiple GPUs to train the model, it is also very simple. Just set the model to a data parallel style model.
After the model is moved to the GPU, a copy will be copied on each GPU, and the data will be divided equally on each GPU for training. The core code is as follows.

```python
# Define model
...

if torch.cuda.device_count()> 1:
    model = nn.DataParallel(model) # Packaged as a parallel style model

# Training model
...
features = features.to(device) # Move data to cuda
labels = labels.to(device) # or labels = labels.cuda() if torch.cuda.is_available() else labels
...
```
<!-- #endregion -->

**The following is a summary of some basic operations related to GPU**


In the Colab notebook: modify -> notebook settings -> hardware accelerator and select GPU

Note: The following code can only be executed correctly on Colab.

You can click the link below to run the example code directly in colab.

"Torch uses gpu training model"

https://colab.research.google.com/drive/1FDmi44-U3TFRCt9MwGn4HIj2SaaWIjHu?usp=sharing

```python
import torch
from torch import nn
```

```python
# 1, view gpu information
if_cuda = torch.cuda.is_available()
print("if_cuda=",if_cuda)

gpu_count = torch.cuda.device_count()
print("gpu_count=",gpu_count)

```

```
if_cuda = True
gpu_count = 1
```

```python
#2, move the tensor between gpu and cpu
tensor = torch.rand((100,100))
tensor_gpu = tensor.to("cuda:0") # or tensor_gpu = tensor.cuda()
print(tensor_gpu.device)
print(tensor_gpu.is_cuda)

tensor_cpu = tensor_gpu.to("cpu") # or tensor_cpu = tensor_gpu.cpu()
print(tensor_cpu.device)

```

```
cuda:0
True
cpu
```

```python
#3, move all the tensors in the model to the gpu
net = nn.Linear(2,1)
print(next(net.parameters()).is_cuda)
net.to("cuda:0") # Put all the parameter tensors in the model to the GPU in turn, note that there is no need to re-assign the value net = net.to("cuda:0")
print(next(net.parameters()).is_cuda)
print(next(net.parameters()).device)
```

```
False
True
cuda:0
```

```python
# 4, create a model that supports multiple GPU data parallelism
linear = nn.Linear(2,1)
print(next(linear.parameters()).device)

model = nn.DataParallel(linear)
print(model.device_ids)
print(next(model.module.parameters()).device)

#Note that when saving parameters, you must specify the parameters to save model.module
torch.save(model.module.state_dict(), "./data/model_parameter.pkl")

linear = nn.Linear(2,1)
linear.load_state_dict(torch.load("./data/model_parameter.pkl"))

```

```
cpu
[0]
cuda:0
```

```python
#5, clear the cuda cache

# This method is very useful when cuda super memory
torch.cuda.empty_cache()

```

```python

```

### One, matrix multiplication example


Next, use the CPU and GPU to make a matrix multiplication, and compare their computational efficiency.

```python
import time
import torch
from torch import nn
```

```python
# Use cpu
a = torch.rand((10000,200))
b = torch.rand((200,10000))
tic = time.time()
c = torch.matmul(a,b)
toc = time.time()

print(toc-tic)
print(a.device)
print(b.device)
```

```
0.6454010009765625
cpu
cpu
```

```python
# Use gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a = torch.rand((10000,200),device = device) #Can specify to create a tensor on the GPU
b = torch.rand((200,10000)) #You can also create a tensor on the CPU and move it to the GPU
b = b.to(device) #or b = b.cuda() if torch.cuda.is_available() else b
tic = time.time()
c = torch.matmul(a,b)
toc = time.time()
print(toc-tic)
print(a.device)
print(b.device)

```

```
0.014541149139404297
cuda:0
cuda:0
```


### Second, linear regression example



The following compares the efficiency of using CPU and GPU to train a linear regression model


**1, use CPU**

```python
# Prepare data
n = 1000000 #sample number

X = 10*torch.rand([n,2])-5.0 #torch.rand is uniform distribution
w0 = torch.tensor([[2.0,-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0.t() + b0 + torch.normal( 0.0,2.0,size = [n,1]) # @ means matrix multiplication, increase normal disturbance
```

```python
# Define model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn_like(w0))
        self.b = nn.Parameter(torch.zeros_like(b0))
    #Forward spread
    def forward(self,x):
        return x@self.w.t() + self.b
        
linear = LinearRegression()

```

```python
# Training model
optimizer = torch.optim.Adam(linear.parameters(),lr = 0.1)
loss_func = nn.MSELoss()

def train(epoches):
    tic = time.time()
    for epoch in range(epoches):
        optimizer.zero_grad()
        Y_pred = linear(X)
        loss = loss_func(Y_pred,Y)
        loss.backward()
        optimizer.step()
        if epoch%50==0:
            print({"epoch":epoch,"loss":loss.item()})
    toc = time.time()
    print("time used:",toc-tic)

train(500)
```

```
{'epoch': 0,'loss': 3.996487855911255}
{'epoch': 50,'loss': 3.9969770908355713}
{'epoch': 100,'loss': 3.9964890480041504}
{'epoch': 150,'loss': 3.996488332748413}
{'epoch': 200,'loss': 3.996488094329834}
{'epoch': 250,'loss': 3.996488332748413}
{'epoch': 300,'loss': 3.996488332748413}
{'epoch': 350,'loss': 3.996488094329834}
{'epoch': 400,'loss': 3.996488332748413}
{'epoch': 450,'loss': 3.996488094329834}
time used: 5.4090576171875
```




**2, use GPU**

```python
# Prepare data
n = 1000000 #sample number

X = 10*torch.rand([n,2])-5.0 #torch.rand is uniform distribution
w0 = torch.tensor([[2.0,-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0.t() + b0 + torch.normal( 0.0,2.0,size = [n,1]) # @ means matrix multiplication, increase normal disturbance

# Move to GPU
print("torch.cuda.is_available() = ",torch.cuda.is_available())
X = X.cuda()
Y = Y.cuda()
print("X.device:",X.device)
print("Y.device:",Y.device)
```

```
torch.cuda.is_available() = True
X.device: cuda:0
Y.device: cuda:0
```

```python
# Define model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn_like(w0))
        self.b = nn.Parameter(torch.zeros_like(b0))
    #Forward spread
    def forward(self,x):
        return x@self.w.t() + self.b
        
linear = LinearRegression()

# Move the model to the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
linear.to(device)

#Check if the model has been moved to the GPU
print("if on cuda:",next(linear.parameters()).is_cuda)

```

```
if on cuda: True
```

```python
# Training model
optimizer = torch.optim.Adam(linear.parameters(),lr = 0.1)
loss_func = nn.MSELoss()

def train(epoches):
    tic = time.time()
    for epoch in range(epoches):
        optimizer.zero_grad()
        Y_pred = linear(X)
        loss = loss_func(Y_pred,Y)
        loss.backward()
        optimizer.step()
        if epoch%50==0:
            print({"epoch":epoch,"loss":loss.item()})
    toc = time.time()
    print("time used:",toc-tic)
    
train(500)
```

```
{'epoch': 0,'loss': 3.9982845783233643}
{'epoch': 50,'loss': 3.998818874359131}
{'epoch': 100,'loss': 3.9982895851135254}
{'epoch': 150,'loss': 3.9982845783233643}
{'epoch': 200,'loss': 3.998284339904785}
{'epoch': 250,'loss': 3.9982845783233643}
{'epoch': 300,'loss': 3.9982845783233643}
{'epoch': 350,'loss': 3.9982845783233643}
{'epoch': 400,'loss': 3.9982845783233643}
{'epoch': 450,'loss': 3.9982845783233643}
time used: 0.4889392852783203
```

```python

```

### Third, torchkeras.Model uses single GPU example


The following demonstrates how to use torchkeras.Model to apply GPU training models.

For the corresponding CPU training model code, see "6-2, 3 methods of training models"

This example only needs to add a line of code to it, and specify device in model.compile.



**1, prepare data**

```python
!pip install -U torchkeras
```

```python
import torch
from torch import nn

import torchvision
from torchvision import transforms

import torchkeras
```

```python
transform = transforms.Compose([transforms.ToTensor()])

ds_train = torchvision.datasets.MNIST(root="./data/minist/",train=True,download=True,transform=transform)
ds_valid = torchvision.datasets.MNIST(root="./data/minist/",train=False,download=True,transform=transform)

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=4)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=128, shuffle=False, num_workers=4)

print(len(ds_train))
print(len(ds_valid))
```

```python
%matplotlib inline
%config InlineBackend.figure_format ='svg'

#View some samples
from matplotlib import pyplot as plt

plt.figure(figsize=(8,8))
for i in range(9):
    img,label = ds_train[i]
    img = torch.squeeze(img)
    ax=plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d"%label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
```

```python

```

**2, define the model**

```python
class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout2d(p = 0.1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,10)]
        )
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

net = CnnModel()
model = torchkeras.Model(net)
model.summary(input_shape=(1,32,32))
```

```
-------------------------------------------------- --------------
        Layer (type) Output Shape Param #
================================================= ==============
            Conv2d-1 [-1, 32, 30, 30] 320
         MaxPool2d-2 [-1, 32, 15, 15] 0
            Conv2d-3 [-1, 64, 11, 11] 51,264
         MaxPool2d-4 [-1, 64, 5, 5] 0
         Dropout2d-5 [-1, 64, 5, 5] 0
 AdaptiveMaxPool2d-6 [-1, 64, 1, 1] 0
           Flatten-7 [-1, 64] 0
            Linear-8 [-1, 32] 2,080
              ReLU-9 [-1, 32] 0
           Linear-10 [-1, 10] 330
================================================= ==============
Total params: 53,994
Trainable params: 53,994
Non-trainable params: 0
-------------------------------------------------- --------------
Input size (MB): 0.003906
Forward/backward pass size (MB): 0.359695
Params size (MB): 0.205971
Estimated Total Size (MB): 0.569572
-------------------------------------------------- --------------
```


**3, training model**

```python
from sklearn.metrics import accuracy_score

def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.cpu().numpy(),y_pred_cls.cpu().numpy())
    # Note that the data must be moved to the cpu first, and then converted into a numpy array

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.compile(loss_func = nn.CrossEntropyLoss(),
             optimizer = torch.optim.Adam(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy},device = device) # Note that the device is specified when compile

dfhistory = model.fit(3,dl_train = dl_train, dl_val=dl_valid, log_step_freq=100)

```

```
Start Training ...

================================================= ==============================2020-06-27 00:24:29
{'step': 100,'loss': 1.063,'accuracy': 0.619}
{'step': 200,'loss': 0.681,'accuracy': 0.764}
{'step': 300,'loss': 0.534,'accuracy': 0.818}
{'step': 400,'loss': 0.458,'accuracy': 0.847}

 +-------+-------+----------+----------+----------- ---+
| epoch | loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+----------- ---+
| 1 | 0.412 | 0.863 | 0.128 | 0.961 |
+-------+-------+----------+----------+----------- ---+

================================================= ==============================2020-06-27 00:24:35
{'step': 100,'loss': 0.147,'accuracy': 0.956}
{'step': 200,'loss': 0.156,'accuracy': 0.954}
{'step': 300,'loss': 0.156,'accuracy': 0.954}
{'step': 400,'loss': 0.157,'accuracy': 0.955}

 +-------+-------+----------+----------+----------- ---+
| epoch | loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+----------- ---+
| 2 | 0.153 | 0.956 | 0.085 | 0.976 |
+-------+-------+----------+----------+----------- ---+

================================================= ==============================2020-06-27 00:24:42
{'step': 100,'loss': 0.126,'accuracy': 0.965}
{'step': 200,'loss': 0.147,'accuracy': 0.96}
{'step': 300,'loss': 0.153,'accuracy': 0.959}
{'step': 400,'loss': 0.147,'accuracy': 0.96}

 +-------+-------+----------+----------+----------- ---+
| epoch | loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+----------- ---+
| 3 | 0.146 | 0.96 | 0.119 | 0.968 |
+-------+-------+----------+----------+----------- ---+

================================================= =============================2020-06-27 00:24:48
Finished Training...
```


**4, evaluation model**

```python
%matplotlib inline
%config InlineBackend.figure_format ='svg'

import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics,'bo--')
    plt.plot(epochs, val_metrics,'ro-')
    plt.title('Training and validation'+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric,'val_'+metric])
    plt.show()
```

```python
plot_metric(dfhistory,"loss")
```

```python
plot_metric(dfhistory,"accuracy")
```

```python
model.evaluate(dl_valid)
```

```
{'val_accuracy': 0.967068829113924,'val_loss': 0.11601964030650598}
```


**5, use model**

```python
model.predict(dl_valid)[0:10]
```

```
tensor([[ -9.2092, 3.1997, 1.4028, -2.7135, -0.7320, -2.0518, -20.4938,
          14.6774, 1.7616, 5.8549],
        [2.8509, 4.9781, 18.0946, 0.0928, -1.6061, -4.1437, 4.8697,
           3.8811, 4.3869, -3.5929],
        [-22.5231, 13.6643, 5.0244, -11.0188, -16.8147, -9.5894, -6.2556,
         -10.5648, -12.1022, -19.4685],
        [23.2670, -12.0711, -7.3968, -8.2715, -1.0915, -12.6050, 8.0444,
         -16.9339, 1.8827, -0.2497],
        [-4.1159, 3.2102, 0.4971, -11.8064, 12.1460, -5.1650, -6.5918,
           1.0088, 0.8362, 2.5132],
        [-26.1764, 15.6251, 6.1191, -12.2424, -13.9725, -10.0540, -7.8669,
          -5.9602, -11.1944, -18.7890],
        [-5.0602, 3.3779, -0.6647, -8.5185, 10.0320, -5.5107, -6.9579,
           2.3811, 0.2542, 3.2860],
        [4.1017, -0.4282, 7.2220, 3.3700, -3.6813, 1.1576, -1.8479,
           0.7450, 3.9768, 6.2640],
        [1.9689, -0.3960, 7.4414, -10.4789, 2.7066, 1.7482, 5.7971,
          -4.5808, 3.0911, -5.1971],
        [-2.9680, -1.2369, -0.0829, -1.8577, 1.9380, -0.8374, -8.2207,
           3.5060, 3.8735, 13.6762]], device='cuda:0')
```


**6, save the model**

```python
# save the model parameters
torch.save(model.state_dict(), "model_parameter.pkl")

model_clone = torchkeras.Model(CnnModel())
model_clone.load_state_dict(torch.load("model_parameter.pkl"))

model_clone.compile(loss_func = nn.CrossEntropyLoss(),
             optimizer = torch.optim.Adam(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy},device = device) # Note that the device is specified when compile

model_clone.evaluate(dl_valid)
```

```
{'val_accuracy': 0.967068829113924,'val_loss': 0.11601964030650598}
```


### Fourth, torchkeras.Model uses multi-GPU examples


Note: The following example needs to be run on a machine with multiple GPUs. If you run on a single GPU machine, it can run through, but in fact a single GPU is used.


**1, prepare data**

```python
import torch
from torch import nn

import torchvision
from torchvision import transforms

import torchkeras
```

```python
transform = transforms.Compose([transforms.ToTensor()])

ds_train = torchvision.datasets.MNIST(root="./data/minist/",train=True,download=True,transform=transform)
ds_valid = torchvision.datasets.MNIST(root="./data/minist/",train=False,download=True,transform=transform)

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=4)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=128, shuffle=False, num_workers=4)

print(len(ds_train))
print(len(ds_valid))
```

**2, define the model**

```python
class CnnModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout2d(p = 0.1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,10)]
        )
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

net = nn.DataParallel(CnnModule()) #Attention this line!!!
model = torchkeras.Model(net)

model.summary(input_shape=(1,32,32))
```

```python

```

**3, training model**

```python
from sklearn.metrics import accuracy_score

def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.cpu().numpy(),y_pred_cls.cpu().numpy())
    # Note that the data must be moved to the cpu first, and then converted into a numpy array

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.compile(loss_func = nn.CrossEntropyLoss(),
             optimizer = torch.optim.Adam(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy},device = device) # Note that the device is specified when compile

dfhistory = model.fit(3,dl_train = dl_train, dl_val=dl_valid, log_step_freq=100)

```

```
Start Training ...

================================================= ==============================2020-06-27 00:24:29
{'step': 100,'loss': 1.063,'accuracy': 0.619}
{'step': 200,'loss': 0.681,'accuracy': 0.764}
{'step': 300,'loss': 0.534,'accuracy': 0.818}
{'step': 400,'loss': 0.458,'accuracy': 0.847}

 +-------+-------+----------+----------+----------- ---+
| epoch | loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+----------- ---+
| 1 | 0.412 | 0.863 | 0.128 | 0.961 |
+-------+-------+----------+----------+----------- ---+

================================================= ==============================2020-06-27 00:24:35
{'step': 100,'loss': 0.147,'accuracy': 0.956}
{'step': 200,'loss': 0.156,'accuracy': 0.954}
{'step': 300,'loss': 0.156,'accuracy': 0.954}
{'step': 400,'loss': 0.157,'accuracy': 0.955}

 +-------+-------+----------+----------+----------- ---+
| epoch | loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+----------- ---+
| 2 | 0.153 | 0.956 | 0.085 | 0.976 |
+-------+-------+----------+----------+----------- ---+

================================================= ==============================2020-06-27 00:24:42
{'step': 100,'loss': 0.126,'accuracy': 0.965}
{'step': 200,'loss': 0.147,'accuracy': 0.96}
{'step': 300,'loss': 0.153,'accuracy': 0.959}
{'step': 400,'loss': 0.147,'accuracy': 0.96}

 +-------+-------+----------+----------+----------- ---+
| epoch | loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+----------- ---+
| 3 | 0.146 | 0.96 | 0.119 | 0.968 |
+-------+-------+----------+----------+----------- ---+

================================================= =============================2020-06-27 00:24:48
Finished Training...
```


**4, evaluation model**

```python
%matplotlib inline
%config InlineBackend.figure_format ='svg'

import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics,'bo--')
    plt.plot(epochs, val_metrics,'ro-')
    plt.title('Training and validation'+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric,'val_'+metric])
    plt.show()
```

```python
plot_metric(dfhistory, "loss")
```

```python
plot_metric(dfhistory,"accuracy")
```

```python
model.evaluate(dl_valid)
```

```
{'val_accuracy': 0.9603441455696202,'val_loss': 0.14203246376371081}
```


**5, use model**

```python
model.predict(dl_valid)[0:10]
```

```
tensor([[ -9.2092, 3.1997, 1.4028, -2.7135, -0.7320, -2.0518, -20.4938,
          14.6774, 1.7616, 5.8549],
        [2.8509, 4.9781, 18.0946, 0.0928, -1.6061, -4.1437, 4.8697,
           3.8811, 4.3869, -3.5929],
        [-22.5231, 13.6643, 5.0244, -11.0188, -16.8147, -9.5894, -6.2556,
         -10.5648, -12.1022, -19.4685],
        [23.2670, -12.0711, -7.3968, -8.2715, -1.0915, -12.6050, 8.0444,
         -16.9339, 1.8827, -0.2497],
        [-4.1159, 3.2102, 0.4971, -11.8064, 12.1460, -5.1650, -6.5918,
           1.0088, 0.8362, 2.5132],
        [-26.1764, 15.6251, 6.1191, -12.2424, -13.9725, -10.0540, -7.8669,
          -5.9602, -11.1944, -18.7890],
        [-5.0602, 3.3779, -0.6647, -8.5185, 10.0320, -5.5107, -6.9579,
           2.3811, 0.2542, 3.2860],
        [4.1017, -0.4282, 7.2220, 3.3700, -3.6813, 1.1576, -1.8479,
           0.7450, 3.9768, 6.2640],
        [1.9689, -0.3960, 7.4414, -10.4789, 2.7066, 1.7482, 5.7971,
          -4.5808, 3.0911, -5.1971],
        [-2.9680, -1.2369, -0.0829, -1.8577, 1.9380, -0.8374, -8.2207,
           3.5060, 3.8735, 13.6762]], device='cuda:0')
```


**6, save the model**

```python
# save the model parameters
torch.save(model.net.module.state_dict(), "model_parameter.pkl")

net_clone = CnnModel()
net_clone.load_state_dict(torch.load("model_parameter.pkl"))

model_clone = torchkeras.Model(net_clone)
model_clone.compile(loss_func = nn.CrossEntropyLoss(),
             optimizer = torch.optim.Adam(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy},device = device)
model_clone.evaluate(dl_valid)

```

```
{'val_accuracy': 0.9603441455696202,'val_loss': 0.14203246376371081}
```

```python

```

### Five, torchkeras.LightModel uses GPU/TPU example



Using torchkeras.LightModel can easily switch the training mode from cpu to single gpu, multiple gpu or even multiple tpu.



**1, prepare data**

```python
import torch
from torch import nn

import torchvision
from torchvision import transforms

import torchkeras
```

```python
transform = transforms.Compose([transforms.ToTensor()])

ds_train = torchvision.datasets.MNIST(root="./data/minist/",train=True,download=True,transform=transform)
ds_valid = torchvision.datasets.MNIST(root="./data/minist/",train=False,download=True,transform=transform)

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=4)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=128, shuffle=False, num_workers=4)

print(len(ds_train))
print(len(ds_valid))
```

**2, define the model**

```python
import torchkeras
import pytorch_lightning as pl

class CnnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout2d(p = 0.1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,10)]
        )
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class Model(torchkeras.LightModel):
    
    #loss,and optional metrics
    def shared_step(self,batch)->dict:
        x, y = batch
        prediction = self(x)
        loss = nn.CrossEntropyLoss()(prediction,y)
        preds = torch.argmax(nn.Softmax(dim=1)(prediction),dim=1).data
        acc = pl.metrics.functional.accuracy(preds, y)
        dic = {"loss":loss,"acc":acc}
        return dic
    
    #optimizer,and optional lr_scheduler
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.0001)
        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
    
pl.seed_everything(1234)
net = CnnNet()
model = Model(net)

torchkeras.summary(model,input_shape=(1,32,32))
print(model)

```

```python

```

**3, training model**

```python
ckpt_cb = pl.callbacks.ModelCheckpoint(monitor='val_loss')

# set gpus=0 will use cpu,
# set gpus=1 will use 1 gpu
# set gpus=2 will use 2gpus
# set gpus = -1 will use all gpus
# you can also set gpus = [0,1] to use the given gpus
# you can even set tpu_cores=2 to use two tpus

trainer = pl.Trainer(max_epochs=10,gpus = 2, callbacks=[ckpt_cb])

trainer.fit(model,dl_train,dl_valid)
```

```
================================================= ==============================2021-01-16 23:13:34
epoch = 0
{'val_loss': 0.0954340249300003,'val_acc': 0.9727057218551636}
{'acc': 0.910403311252594,'loss': 0.27809813618659973}

================================================= ==============================2021-01-16 23:15:02
epoch = 1
{'val_loss': 0.06748798489570618,'val_acc': 0.9809137582778931}
{'acc': 0.9663013219833374,'loss': 0.10915637016296387}

================================================= ==============================2021-01-16 23:16:34
epoch = 2
{'val_loss': 0.06344369053840637,'val_acc': 0.980320394039154}
{'acc': 0.9712153673171997,'loss': 0.09515620768070221}

================================================= ==============================2021-01-16 23:18:05
epoch = 3
{'val_loss': 0.08105307072401047,'val_acc': 0.977155864238739}
{'acc': 0.9747745990753174,'loss': 0.08337805420160294}

================================================= ==============================2021-01-16 23:19:38
epoch = 4
{'val_loss': 0.06881670653820038,'val_acc': 0.9798259735107422}
{'acc': 0.9764847159385681,'loss': 0.08077647536993027}

================================================= ==============================2021-01-16 23:21:11
epoch = 5
{'val_loss': 0.07127966731786728,'val_acc': 0.980320394039154}
{'acc': 0.9758350849151611,'loss': 0.08572731912136078}

================================================= ==============================2021-01-16 23:22:41
epoch = 6
{'val_loss': 0.1256944239139557,'val_acc': 0.9672666192054749}
{'acc': 0.978233814239502,'loss': 0.07292930781841278}

================================================= ==============================2021-01-16 23:24:05
epoch = 7
{'val_loss': 0.08458385616540909,'val_acc': 0.9767602682113647}
{'acc': 0.9790666699409485,'loss': 0.0768343135714531}

================================================= ==============================2021-01-16 23:25:32
epoch = 8
{'val_loss': 0.06721501052379608,'val_acc': 0.983188271522522}
{'acc': 0.9786669015884399,'loss': 0.07818026840686798}

================================================= ==============================2021-01-16 23:26:56
epoch = 9
{'val_loss': 0.06671519577503204,'val_acc': 0.9839794039726257}
{'acc': 0.9826259613037109,'loss': 0.06241251528263092}
```

```python

```

**4, evaluation model**

```python
import pandas as pd

history = model.history
dfhistory = pd.DataFrame(history)
dfhistory
```

```python
%matplotlib inline
%config InlineBackend.figure_format ='svg'

import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics,'bo--')
    plt.plot(epochs, val_metrics,'ro-')
    plt.title('Training and validation'+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric,'val_'+metric])
    plt.show()
```

```python
plot_metric(dfhistory,"loss")
```

```python
plot_metric(dfhistory,"acc")
```

```python
results = trainer.test(model, test_dataloaders=dl_valid, verbose = False)
print(results[0])
```

```
{'test_loss': 0.005034677684307098,'test_acc': 1.0}
```

```python

```

**5, use model**

```python
def predict(model,dl):
    model.eval()
    preds = torch.cat([model.forward(t[0].to(model.device)) for t in dl])
    
    result = torch.argmax(nn.Softmax(dim=1)(preds),dim=1).data
    return(result.data)

result = predict(model,dl_valid)
result
```

```
tensor([7, 2, 1, ..., 4, 5, 6])
```

```python

```

**6, save the model**

```python
print(ckpt_cb.best_model_score)
model.load_from_checkpoint(ckpt_cb.best_model_path)

best_net = model.net
torch.save(best_net.state_dict(),"./data/net.pt")
```

```python
net_clone = CnnNet()
net_clone.load_state_dict(torch.load("./data/net.pt"))
model_clone = Model(net_clone)
trainer = pl.Trainer()
result = trainer.test(model_clone,test_dataloaders=dl_valid, verbose = False)

print(result)
```

```python

```

**If this book is helpful to you and want to encourage the author, remember to add a star⭐️ to this project and share it with your friends 😊!**

If you need to further communicate with the author on the understanding of the content of this book, please leave a message under the public account "Algorithm Food House". The author has limited time and energy and will respond as appropriate.

You can also reply to keywords in the background of the official account: **Add group**, join the reader exchange group and discuss with you.

![算法美食屋logo.png](./data/算法美食屋二维码.jpg)
