# 3-3, high-level API demonstration

Pytorch does not have an official high-level API, and generally requires users to implement training loops, verification loops, and prediction loops.

The author encapsulated Pytorch's nn.Module by imitating the function of tf.keras.Model, and designed the torchkeras.Model class,

Implemented fit, validate, predict, and summary methods, which are equivalent to user-defined high-level APIs.

And demonstrated the use of it to implement linear regression models.

In addition, the author also encapsulates another implementation of the Keras-like interface by borrowing the function of pytorch_lightning, namely the torchkeras.LightModel class.

And demonstrated the use of it to realize the DNN two classification model.



```python
import os
import datetime
from torchkeras import Model, summary

#Print Time
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)

#Mac system pytorch and matplotlib running at the same time in jupyter need to change environment variables
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

```

### One, linear regression model


In this example, we implement the linear regression model by inheriting the torchkeras.Model model interface.


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

![](./data/3-3-回归数据可视化.png)

```python
#Build input data pipeline
ds = TensorDataset(X,Y)
ds_train,ds_valid = torch.utils.data.random_split(ds,[int(400*0.7),400-int(400*0.7)])
dl_train = DataLoader(ds_train,batch_size = 10,shuffle=True,num_workers=2)
dl_valid = DataLoader(ds_valid,batch_size = 10,num_workers=2)

```

```python

```

**2, define the model**

```python
# Inherit the user-defined model
from torchkeras import Model
class LinearRegression(Model):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(2,1)
    
    def forward(self,x):
        return self.fc(x)

model = LinearRegression()

```

```python
model.summary(input_shape = (2,))
```

```
-------------------------------------------------- --------------
        Layer (type) Output Shape Param #
================================================= ==============
            Linear-1 [-1, 1] 3
================================================= ==============
Total params: 3
Trainable params: 3
Non-trainable params: 0
-------------------------------------------------- --------------
Input size (MB): 0.000008
Forward/backward pass size (MB): 0.000008
Params size (MB): 0.000011
Estimated Total Size (MB): 0.000027
-------------------------------------------------- --------------
```

```python

```

**3, training model**

```python
### Use fit method for training

def mean_absolute_error(y_pred,y_true):
    return torch.mean(torch.abs(y_pred-y_true))

def mean_absolute_percent_error(y_pred,y_true):
    absolute_percent_error = (torch.abs(y_pred-y_true)+1e-7)/(torch.abs(y_true)+1e-7)
    return torch.mean(absolute_percent_error)

model.compile(loss_func = nn.MSELoss(),
              optimizer = torch.optim.Adam(model.parameters(),lr = 0.01),
              metrics_dict={"mae":mean_absolute_error,"mape":mean_absolute_percent_error})

dfhistory = model.fit(200,dl_train = dl_train, dl_val = dl_valid,log_step_freq = 20)

```

```
Start Training ...

================================================= ==============================2020-07-05 23:07:25
{'step': 20,'loss': 226.768,'mae': 12.198,'mape': 1.212}

 +-------+---------+-------+-------+----------+---- -----+----------+
| epoch | loss | mae | mape | val_loss | val_mae | val_mape |
+-------+---------+-------+-------+----------+---- -----+----------+
| 1 | 230.773 | 12.41 | 1.394 | 223.262 | 12.582 | 1.095 |
+-------+---------+-------+-------+----------+---- -----+----------+

================================================= ==============================2020-07-05 23:07:26
{'step': 20,'loss': 200.964,'mae': 11.584,'mape': 1.382}

 +-------+---------+--------+------+----------+---- -----+----------+
| epoch | loss | mae | mape | val_loss | val_mae | val_mape |
+-------+---------+--------+------+----------+---- -----+----------+
| 2 | 206.238 | 11.759 | 1.26 | 199.669 | 11.895 | 1.012 |
+-------+---------+--------+------+----------+---- -----+----------+

================================================= ==============================2020-07-05 23:07:26
{'step': 20,'loss': 188.247,'mae': 11.387,'mape': 1.172}

 +-------+---------+--------+-------+----------+--- ------+----------+
| epoch | loss | mae | mape | val_loss | val_mae | val_mape |
+-------+---------+--------+-------+----------+--- ------+----------+
| 3 | 185.185 | 11.177 | 1.189 | 178.112 | 11.24 | 0.952 |
+-------+---------+--------+-------+----------+--- ------+----------+
================================================= ==============================2020-07-05 23:07:59
{'step': 20,'loss': 4.14,'mae': 1.677,'mape': 1.845}

 +-------+-------+-------+-------+----------+------ ---+----------+
| epoch | loss | mae | mape | val_loss | val_mae | val_mape |
+-------+-------+-------+-------+----------+------ ---+----------+
| 199 | 4.335 | 1.707 | 1.441 | 3.741 | 1.533 | 0.359 |
+-------+-------+-------+-------+----------+------ ---+----------+

================================================= ==============================2020-07-05 23:07:59
{'step': 20,'loss': 4.653,'mae': 1.775,'mape': 0.679}

 +-------+------+-------+-------+----------+------- --+----------+
| epoch | loss | mae | mape | val_loss | val_mae | val_mape |
+-------+------+-------+-------+----------+------- --+----------+
| 200 | 4.37 | 1.718 | 1.394 | 3.749 | 1.534 | 0.363 |
+-------+------+-------+-------+----------+------- --+----------+

================================================= ==============================2020-07-05 23:07:59
Finished Training...
```

```python
# Result visualization

%matplotlib inline
%config InlineBackend.figure_format ='svg'

w,b = model.state_dict()["fc.weight"],model.state_dict()["fc.bias"]

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

**4, evaluation model**

```python
dfhistory.tail()
```
![](./data/3-3-dfhistory.png)

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
![](./data/3-3-loss曲线.png)

```python
plot_metric(dfhistory,"mape")
```

![](./data/3-3-mape曲线.png)

```python
# Evaluation
model.evaluate(dl_valid)
```

```
{'val_loss': 3.749117374420166,
 'val_mae': 1.5336137612660725,
 'val_mape': 0.36319838215907413}
```

```python

```

**5, use model**

```python
# Forecast
dl = DataLoader(TensorDataset(X))
model.predict(dl)[0:10]
```

```
tensor([[ 3.9212],
        [8.6336],
        [6.1982],
        [6.1212],
        [-5.0974],
        [-6.3183],
        [4.6588],
        [5.5349],
        [11.9106],
        [24.6937]])
```

```python
# Forecast
model.predict(dl_valid)[0:10]
```

```
tensor([[ 2.8368],
        [16.2797],
        [2.3135],
        [9.5395],
        [16.4363],
        [10.0742],
        [15.0864],
        [12.9775],
        [21.8568],
        [21.8226]])
```

```python

```

### Two, DNN two classification model


In this example, we implement the DNN binary classification model by inheriting the torchkeras.LightModel model interface.



**1, prepare data**

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset
import torchkeras
import pytorch_lightning as pl
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

![](./data/3-3-分类数据可视化.png)

```python
ds = TensorDataset(X,Y)

ds_train,ds_valid = torch.utils.data.random_split(ds,[int(len(ds)*0.7),len(ds)-int(len(ds)*0.7)])
dl_train = DataLoader(ds_train,batch_size = 100,shuffle=True,num_workers=2)
dl_valid = DataLoader(ds_valid,batch_size = 100,num_workers=2)

```

```python

```

**2, define the model**

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,8)
        self.fc3 = nn.Linear(8,1)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = nn.Sigmoid()(self.fc3(x))
        return y
        
class Model(torchkeras.LightModel):
    
    #loss,and optional metrics
    def shared_step(self,batch)->dict:
        x, y = batch
        prediction = self(x)
        loss = nn.BCELoss()(prediction,y)
        preds = torch.where(prediction>0.5,torch.ones_like(prediction),torch.zeros_like(prediction))
        acc = pl.metrics.functional.accuracy(preds, y)
        # attention: there must be a key of "loss" in the returned dict
        dic = {"loss":loss,"acc":acc}
        return dic
    
    #optimizer,and optional lr_scheduler
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.0001)
        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
    
pl.seed_everything(1234)
net = Net()
model = Model(net)


torchkeras.summary(model,input_shape =(2,))


```

```
-------------------------------------------------- --------------
        Layer (type) Output Shape Param #
================================================= ==============
            Linear-1 [-1, 4] 12
            Linear-2 [-1, 8] 40
            Linear-3 [-1, 1] 9
================================================= ==============
Total params: 61
Trainable params: 61
Non-trainable params: 0
-------------------------------------------------- --------------
Input size (MB): 0.000008
Forward/backward pass size (MB): 0.000099
Params size (MB): 0.000233
Estimated Total Size (MB): 0.000340
-------------------------------------------------- --------------
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

trainer = pl.Trainer(max_epochs=100,gpus = 0, callbacks=[ckpt_cb])

trainer.fit(model,dl_train,dl_valid)

```

```
================================================= ==============================2021-01-16 23:41:38
epoch = 0
{'val_loss': 0.6706896424293518,'val_acc': 0.5558333396911621}
{'acc': 0.5157142281532288,'loss': 0.6820458769798279}

================================================= ==============================2021-01-16 23:41:39
epoch = 1
{'val_loss': 0.653035581111908,'val_acc': 0.5708333849906921}
{'acc': 0.5457143783569336,'loss': 0.6677185297012329}

================================================= ==============================2021-01-16 23:41:40
epoch = 2
{'val_loss': 0.6122683882713318,'val_acc': 0.6533333659172058}
{'acc': 0.6132143139839172,'loss': 0.6375874876976013}

================================================= ==============================2021-01-16 23:41:40
epoch = 3
{'val_loss': 0.5168119668960571,'val_acc': 0.7708333134651184}
{'acc': 0.6842857003211975,'loss': 0.574131190776825}

================================================= ==============================2021-01-16 23:41:41
epoch = 4
{'val_loss': 0.3789764940738678,'val_acc': 0.8766666054725647}
{'acc': 0.7900000214576721,'loss': 0.4608381390571594}

================================================= ==============================2021-01-16 23:41:41
epoch = 5
{'val_loss': 0.2496153712272644,'val_acc': 0.9208332896232605}
{'acc': 0.8982142806053162,'loss': 0.3223666250705719}

================================================= ==============================2021-01-16 23:41:42
epoch = 6
{'val_loss': 0.21876734495162964,'val_acc': 0.9124999642372131}
{'acc': 0.908214271068573,'loss': 0.24333418905735016}

================================================= ==============================2021-01-16 23:41:43
epoch = 7
{'val_loss': 0.19420616328716278,'val_acc': 0.9266666769981384}
{'acc': 0.9132143259048462,'loss': 0.2207658737897873}

================================================= ==============================2021-01-16 23:41:44
epoch = 8
{'val_loss': 0.1835813671350479,'val_acc': 0.9225000739097595}
{'acc': 0.9185715317726135,'loss': 0.20826208591461182}

================================================= ==============================2021-01-16 23:41:45
epoch = 9
{'val_loss': 0.17465434968471527,'val_acc': 0.9300000071525574}
{'acc': 0.9189285039901733,'loss': 0.20436131954193115}
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

![](./data/3-3-分类结果可视化.png)


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

![](./data/3-3-分类loss曲线.png)

```python
plot_metric(dfhistory,"acc")
```

![](./data/3-3-分类acc曲线.png)

```python
results = trainer.test(model, test_dataloaders=dl_valid, verbose = False)
print(results[0])
```

```
{'test_loss': 0.18403057754039764,'test_acc': 0.949999988079071}
```

```python

```

**5, use model**

```python
def predict(model,dl):
    model.eval()
    prediction = torch.cat([model.forward(t[0].to(model.device)) for t in dl])
    result = torch.where(prediction>0.5,torch.ones_like(prediction),torch.zeros_like(prediction))
    return(result.data)

result = predict(model,dl_valid)

result
```

```
tensor([[0.],
        [0.],
        [0.],
        ...,
        [1.],
        [1.],
        [1.]])

```

```python

```

**If this book is helpful to you and want to encourage the author, remember to add a star to this project, and share it with your friends 😊!**

If you need to further communicate with the author on the understanding of the content of this book, please leave a message under the public account "Algorithm Food House". The author has limited time and energy and will respond as appropriate.

You can also reply to keywords in the background of the official account: **Add group**, join the reader exchange group and discuss with you.


![算法美食屋logo.png](./data/算法美食屋二维码.jpg)

