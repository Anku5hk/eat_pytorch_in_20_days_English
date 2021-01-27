# 3-2, Intermediate API demonstration

The following example uses Pytorch's mid-level API to implement linear regression models and DNN binary classification models.

Pytorch's mid-level API mainly includes various model layers, loss functions, optimizers, data pipelines, etc.

```python
import os
import datetime

#Print Time
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)

#Mac system pytorch and matplotlib running at the same time in jupyter need to change environment variables
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

```

```python

```

### One, linear regression model


**1, prepare data**

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset

#Number of samples
n = 400

# Generate test data set
X = 10*torch.rand([n,2])-5.0 #torch.rand is uniform distribution
w0 = torch.tensor([[2.0],[-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0 + b0 + torch.normal( 0.0,2.0,size = [n,1]) # @ means matrix multiplication, increase normal disturbance

```

```python
# data visualization

%matplotlib inline
%config InlineBackend.figure_format ='svg'

plt.figure(figsize = (12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0],Y[:,0], c = "b",label = "samples")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y",rotation = 0)

ax2 = plt.subplot(122)
ax2.scatter(X[:,1],Y[:,0], c = "g",label = "samples")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y",rotation = 0)
plt.show()

```

![](./data/3-2-线性回归数据可视化.png)

```python
#Build input data pipeline
ds = TensorDataset(X,Y)
dl = DataLoader(ds,batch_size = 10,shuffle=True,num_workers=2)

```

```python

```

**2, define the model**

```python
model = nn.Linear(2,1) #Linear layer

model.loss_func = nn.MSELoss()
model.optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

```

```python

```

**3, training model**

```python
def train_step(model, features, labels):
    
    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    loss.backward()
    model.optimizer.step()
    model.optimizer.zero_grad()
    return loss.item()

# Test the effect of train_step
features,labels = next(iter(dl))
train_step(model,features,labels)


```

```
269.98016357421875
```

```python
def train_model(model,epochs):
    for epoch in range(1,epochs+1):
        for features, labels in dl:
            loss = train_step(model,features,labels)
        if epoch%50==0:
            printbar()
            w = model.state_dict()["weight"]
            b = model.state_dict()["bias"]
            print("epoch =",epoch,"loss = ",loss)
            print("w =",w)
            print("b =",b)
train_model(model,epochs = 200)

```

```
================================================= =============================2020-07-05 22:51:53
epoch = 50 loss = 3.0177409648895264
w = tensor([[ 1.9315, -2.9573]])
b = tensor([9.9625])

================================================= =============================2020-07-05 22:51:57
epoch = 100 loss = 2.1144354343414307
w = tensor([[ 1.9760, -2.9398]])
b = tensor([9.9428])

================================================= ==============================2020-07-05 22:52:01
epoch = 150 loss = 3.290461778640747
w = tensor([[ 2.1075, -2.9509]])
b = tensor([9.9599])

================================================= ==============================2020-07-05 22:52:06
epoch = 200 loss = 3.047853469848633
w = tensor([[ 2.1134, -2.9306]])
b = tensor([9.9722])
```

```python

```

```python
# Result visualization

%matplotlib inline
%config InlineBackend.figure_format ='svg'

w,b = model.state_dict()["weight"],model.state_dict()["bias"]

plt.figure(figsize = (12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0],Y[:,0], c = "b",label = "samples")
ax1.plot(X[:,0],w[0,0]*X[:,0]+b[0],"-r",linewidth = 5.0,label = "model")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y",rotation = 0)



ax2 = plt.subplot(122)
ax2.scatter(X[:,1],Y[:,0], c = "g",label = "samples")
ax2.plot(X[:,1],w[0,1]*X[:,1]+b[0],"-r",linewidth = 5.0,label = "model")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y",rotation = 0)

plt.show()

```

![](./data/3-2-回归结果可视化.png)

```python

```

### Two, DNN two classification model

```python

```

**1, prepare data**

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset
%matplotlib inline
%config InlineBackend.figure_format ='svg'

#Number of positive and negative samples
n_positive,n_negative = 2000,2000

#Generate positive samples, small circle distribution
r_p = 5.0 + torch.normal(0.0,1.0,size = [n_positive,1])
theta_p = 2*np.pi*torch.rand([n_positive,1])
Xp = torch.cat([r_p*torch.cos(theta_p),r_p*torch.sin(theta_p)],axis = 1)
Yp = torch.ones_like(r_p)

#Generate negative samples, large circle distribution
r_n = 8.0 + torch.normal(0.0,1.0,size = [n_negative,1])
theta_n = 2*np.pi*torch.rand([n_negative,1])
Xn = torch.cat([r_n*torch.cos(theta_n),r_n*torch.sin(theta_n)],axis = 1)
Yn = torch.zeros_like(r_n)

#Summary sample
X = torch.cat([Xp,Xn],axis = 0)
Y = torch.cat([Yp,Yn],axis = 0)


#Visualization
plt.figure(figsize = (6,6))
plt.scatter(Xp[:,0],Xp[:,1],c = "r")
plt.scatter(Xn[:,0],Xn[:,1],c = "g")
plt.legend(["positive","negative"]);

```
![](./data/3-2-分类数据可视化.png)

```python
#Build input data pipeline
ds = TensorDataset(X,Y)
dl = DataLoader(ds,batch_size = 10,shuffle=True,num_workers=2)


```

**2, define the model**

```python
class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,8)
        self.fc3 = nn.Linear(8,1)

    # Forward spread
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = nn.Sigmoid()(self.fc3(x))
        return y
    
    # Loss function
    def loss_func(self,y_pred,y_true):
        return nn.BCELoss()(y_pred,y_true)
    
    # Evaluation function (accuracy rate)
    def metric_func(self,y_pred,y_true):
        y_pred = torch.where(y_pred>0.5,torch.ones_like(y_pred,dtype = torch.float32),
                          torch.zeros_like(y_pred,dtype = torch.float32))
        acc = torch.mean(1-torch.abs(y_true-y_pred))
        return acc
    
    # Optimizer
    @property
    def optimizer(self):
        return torch.optim.Adam(self.parameters(),lr = 0.001)
    
model = DNNModel()

```

```python
# Test model structure
(features,labels) = next(iter(dl))
predictions = model(features)

loss = model.loss_func(predictions,labels)
metric = model.metric_func(predictions,labels)

print("init loss:",loss.item())
print("init metric:",metric.item())

```

```
init loss: 0.7065666913986206
init metric: 0.6000000238418579
```

```python

```

**3, training model**

```python
def train_step(model, features, labels):
    
    # Forward propagation for loss
    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)
    
    # Backpropagation for gradient
    loss.backward()
    
    # Update model parameters
    model.optimizer.step()
    model.optimizer.zero_grad()
    
    return loss.item(),metric.item()

# Test the effect of train_step
features,labels = next(iter(dl))
train_step(model,features,labels)

```

```
(0.6048880815505981, 0.699999988079071)
```

```python
def train_model(model,epochs):
    for epoch in range(1,epochs+1):
        loss_list,metric_list = [],[]
        for features, labels in dl:
            lossi,metrici = train_step(model,features,labels)
            loss_list.append(lossi)
            metric_list.append(metrici)
        loss = np.mean(loss_list)
        metric = np.mean(metric_list)

        if epoch%100==0:
            printbar()
            print("epoch =",epoch,"loss = ",loss,"metric = ",metric)
        
train_model(model,epochs = 300)
```

```
================================================= ==============================2020-07-05 22:56:38
epoch = 100 loss = 0.23532892110607917 metric = 0.934749992787838

================================================= ==============================2020-07-05 22:58:18
epoch = 200 loss = 0.24743918558603128 metric = 0.934999993443489

================================================= ==============================2020-07-05 22:59:56
epoch = 300 loss = 0.2936080049697884 metric = 0.931499992609024
```

```python

```

```python
# Result visualization
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize = (12,5))
ax1.scatter(Xp[:,0],Xp[:,1], c="r")
ax1.scatter(Xn[:,0],Xn[:,1],c = "g")
ax1.legend(["positive","negative"]);
ax1.set_title("y_true");

Xp_pred = X[torch.squeeze(model.forward(X)>=0.5)]
Xn_pred = X[torch.squeeze(model.forward(X)<0.5)]

ax2.scatter(Xp_pred[:,0],Xp_pred[:,1],c = "r")
ax2.scatter(Xn_pred[:,0],Xn_pred[:,1],c = "g")
ax2.legend(["positive","negative"]);
ax2.set_title("y_pred");

```

![](./data/3-2-分类结果可视化.png)

```python

```

**If this book is helpful to you and want to encourage the author, remember to add a star to this project, and share it with your friends 😊!**

If you need to further communicate with the author on the understanding of the content of this book, please leave a message under the public account "Algorithm Food House". The author has limited time and energy and will respond as appropriate.

You can also reply to keywords in the background of the official account: **Add group**, join the reader exchange group and discuss with you.

![算法美食屋logo.png](./data/算法美食屋二维码.jpg)
