# 5-1, Dataset and DataLoader

Pytorch usually uses two tool classes, Dataset and DataLoader, to build data pipelines.

Dataset defines the content of the data set, which is equivalent to a data structure similar to a list, has a certain length, and can use the index to get the elements in the data set.

The DataLoader defines a method for loading data sets in batches. It is an iterable object that implements the `__iter__` method, and outputs a batch of data for each iteration.

DataLoader can control the size of the batch, the sampling method of the elements in the batch, and the method of sorting the batch results into the input form required by the model, and can use multiple processes to read data.

In most cases, users only need to implement the `__len__` method and `__getitem__` method of the Dataset, and then they can easily build their own data set and load it with the default data pipeline.


```python

```

### One, Overview of Dataset and DataLoader


**1, steps to obtain a batch data**


Let us consider the steps required to obtain a batch of data from a data set.

(Assuming that the features and labels of the data set are expressed as tensors `X` and `Y`, the data set can be expressed as `(X,Y)`, assuming the batch size is `m`)

1. First, we have to determine the length of the data set `n`.

The result is similar: `n = 1000`.

2. Then we sample the number of `m` (batch size) from the range of `0` to `n-1`.

Assuming `m=4`, the result obtained is a list, similar to: `indices = [1,4,8,9]`

3. Then we get the elements of the subscript corresponding to the number of `m` from the data set.

The result is a list of tuples, similar to: `samples = [(X[1],Y[1]),(X[4],Y[4]),(X[8],Y[8] ),(X[9],Y[9])]`

4. Finally, we organize the result into two tensors as output.

The result is two tensors, similar to `batch = (features,labels)`,

Where `features = torch.stack([X[1],X[4],X[8],X[9]])`

`labels = torch.stack([Y[1],Y[4],Y[8],Y[9]])`


```python

```

**2, the division of labor between Dataset and DataLoader**


The first step above to determine the length of the data set is implemented by the `__len__` method of Dataset.

In the second step, the method of sampling the number of `m` from the range of `0` to `n-1` is specified by the `sampler` and `batch_sampler` parameters of DataLoader.

The `sampler` parameter specifies the sampling method of a single element, and generally does not need to be set by the user. By default, the program uses random sampling when the DataLoader parameter `shuffle=True`, and uses sequential sampling when `shuffle=False`.

The `batch_sampler` parameter organizes multiple sampled elements into a list, which generally does not need to be set by the user. The default method will discard the last batch of the data set whose length is not divisible by the batch size when the DataLoader parameter `drop_last=True`. Keep the last batch when drop_last=False`.

The core logic of the third step is to get the elements in the data set according to the subscripts, which is implemented by the `__getitem__` method of the Dataset.

The logic of the fourth step is specified by the parameter `collate_fn` of the DataLoader. Under normal circumstances, there is no need for user settings.

```python

```

**3, the main interface of Dataset and DataLoader**


The following is the core interface logic pseudo code of Dataset and DataLoader, which is not completely consistent with the source code.

```python
import torch
class Dataset(object):
    def __init__(self):
        pass
    
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self,index):
        raise NotImplementedError
        

class DataLoader(object):
    def __init__(self,dataset,batch_size,collate_fn,shuffle = True,drop_last = False):
        self.dataset = dataset
        self.sampler =torch.utils.data.RandomSampler if shuffle else \
           torch.utils.data.SequentialSampler
        self.batch_sampler = torch.utils.data.BatchSampler
        self.sample_iter = self.batch_sampler(
            self.sampler(range(len(dataset))),
            batch_size = batch_size,drop_last = drop_last)
        
    def __next__(self):
        indices = next(self.sample_iter)
        batch = self.collate_fn([self.dataset[i] for i in indices])
        return batch
    
```

```python

```

### Second, use Dataset to create a data set

<!-- #region -->
Dataset commonly used methods to create data sets are:

* Use torch.utils.data.TensorDataset to create a data set based on Tensor (numpy array, Pandas DataFrame needs to be converted to Tensor).

* Use torchvision.datasets.ImageFolder to create image datasets based on the image catalog.

* Inherit torch.utils.data.Dataset to create a custom data set.


In addition, you can also pass

* torch.utils.data.random_split splits a data set into multiple parts, which is often used to split training set, validation set and test set.

* Call the addition operator (`+`) of Dataset to merge multiple data sets into one data set.
<!-- #endregion -->

**1, create a data set based on Tensor**

```python
import numpy as np
import torch
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split

```

```python
# Create a data set based on Tensor

from sklearn import datasets
iris = datasets.load_iris()
ds_iris = TensorDataset(torch.tensor(iris.data),torch.tensor(iris.target))

# Split into training set and prediction set
n_train = int(len(ds_iris)*0.8)
n_valid = len(ds_iris)-n_train
ds_train,ds_valid = random_split(ds_iris,[n_train,n_valid])

print(type(ds_iris))
print(type(ds_train))

```

```python
# Use DataLoader to load data sets
dl_train,dl_valid = DataLoader(ds_train,batch_size = 8),DataLoader(ds_valid,batch_size = 8)

for features,labels in dl_train:
    print(features,labels)
    break
```

```python
# Demonstrate the combined effect of the addition operator (`+`)

ds_data = ds_train + ds_valid

print('len(ds_train) =',len(ds_train))
print('len(ds_valid) =',len(ds_valid))
print('len(ds_train+ds_valid) =',len(ds_data))

print(type(ds_data))

```

```python

```

**2, create a picture data set according to the picture directory**

```python
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms,datasets

```

```python
#Demonstrate some common image enhancement operations
```

```python
from PIL import Image
img = Image.open('./data/cat.jpeg')
img
```

![](./data/5-1-傻乎乎.png)

```python
# Random value flip
transforms.RandomVerticalFlip()(img)
```

![](./data/5-1-翻转.png)
```python
#Random rotation
transforms.RandomRotation(45)(img)
```


![](./data/5-1-旋转.png)

```python
# Define image enhancement operations

transform_train = transforms.Compose([
   transforms.RandomHorizontalFlip(), #Random horizontal flip
   transforms.RandomVerticalFlip(), #Random vertical flip
   transforms.RandomRotation(45), #Randomly rotate within 45 degrees
   transforms.ToTensor() #convert to tensor
  ]
)

transform_valid = transforms.Compose([
    transforms.ToTensor()
  ]
)

```

```python
# Create a data set based on the picture directory
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
# Use DataLoader to load data sets

dl_train = DataLoader(ds_train, batch_size = 50, shuffle = True, num_workers=3)
dl_valid = DataLoader(ds_valid, batch_size = 50, shuffle = True, num_workers=3)
```

```python
for features,labels in dl_train:
    print(features.shape)
    print(labels.shape)
    break
```

```
torch.Size([50, 3, 32, 32])
torch.Size([50, 1])
```

```python

```

**3, create a custom data set**


Next, create a custom data set for imdb text classification tasks by inheriting the Dataset class.

The general idea is as follows: First, construct a dictionary for word segmentation of the training set text. Then convert the training set text and test set text data into token word encoding.

Then the training set data and the test set data converted into word codes are divided into multiple files according to the samples, and one file represents one sample.

Finally, we can obtain the sample content corresponding to the serial number according to the file name list to build the Dataset data set.


```python
import numpy as np
import pandas as pd
from collections import OrderedDict
import re,string

MAX_WORDS = 10000 # Only consider the most frequent 10000 words
MAX_LEN = 200 # Each sample retains the length of 200 words
BATCH_SIZE = 20

train_data_path ='data/imdb/train.tsv'
test_data_path ='data/imdb/test.tsv'
train_token_path ='data/imdb/train_token.tsv'
test_token_path ='data/imdb/test_token.tsv'
train_samples_path ='data/imdb/train_samples/'
test_samples_path ='data/imdb/test_samples/'
```

First, we build a dictionary and keep the most frequent MAX_WORDS words.

```python
##Build a dictionary

word_count_dict = {}

#Cleaning text
def clean_text(text):
    lowercase = text.lower().replace("\n"," ")
    stripped_html = re.sub('<br />', '',lowercase)
    cleaned_punctuation = re.sub('[%s]'%re.escape(string.punctuation),'',stripped_html)
    return cleaned_punctuation

with open(train_data_path,"r",encoding ='utf-8') as f:
    for line in f:
        label,text = line.split("\t")
        cleaned_text = clean_text(text)
        for word in cleaned_text.split(" "):
            word_count_dict[word] = word_count_dict.get(word,0)+1

df_word_dict = pd.DataFrame(pd.Series(word_count_dict,name = "count"))
df_word_dict = df_word_dict.sort_values(by = "count",ascending=False)

df_word_dict = df_word_dict[0:MAX_WORDS-2] #
df_word_dict["word_id"] = range(2,MAX_WORDS) #Numbers 0 and 1 are reserved for unknown words <unkown> and padding <padding> respectively

word_id_dict = df_word_dict["word_id"].to_dict()

df_word_dict.head(10)

```

![](./data/5-1-词典.png)


Then we use the built dictionary to convert the text into a token serial number.

```python
#Conversion token

# Fill text
def pad(data_list,pad_length):
    padded_list = data_list.copy()
    if len(data_list)> pad_length:
         padded_list = data_list[-pad_length:]
    if len(data_list)< pad_length:
         padded_list = [1]*(pad_length-len(data_list))+data_list
    return padded_list

def text_to_token(text_file,token_file):
    with open(text_file,"r",encoding ='utf-8') as fin,\
      open(token_file,"w",encoding ='utf-8') as fout:
        for line in fin:
            label,text = line.split("\t")
            cleaned_text = clean_text(text)
            word_token_list = [word_id_dict.get(word, 0) for word in cleaned_text.split(" ")]
            pad_list = pad(word_token_list,MAX_LEN)
            out_line = label+"\t"+" ".join([str(x) for x in pad_list])
            fout.write(out_line+"\n")
        
text_to_token(train_data_path,train_token_path)
text_to_token(test_data_path,test_token_path)

```

Next, the token text is divided into samples, and each file stores the data of a sample.

```python
# Split sample
import os

if not os.path.exists(train_samples_path):
    os.mkdir(train_samples_path)
    
if not os.path.exists(test_samples_path):
    os.mkdir(test_samples_path)
    
    
def split_samples(token_path,samples_dir):
    with open(token_path,"r",encoding ='utf-8') as fin:
        i = 0
        for line in fin:
            with open(samples_dir+"%d.txt"%i,"w",encoding = "utf-8") as fout:
                fout.write(line)
            i = i+1

split_samples(train_token_path,train_samples_path)
split_samples(test_token_path,test_samples_path)
```

```python
print(os.listdir(train_samples_path)[0:100])
```

```
['11303.txt', '3644.txt', '19987.txt', '18441.txt', '5235.txt', '17772.txt', '1053.txt', '13514.txt', ' 8711.txt', '15165.txt', '7422.txt', '8077.txt', '15603.txt', '7344.txt', '1735.txt', '13272.txt', '9369. txt', '18327.txt', '5553.txt', '17014.txt', '4895.txt', '11465.txt', '3122.txt', '19039.txt', '5547.txt' , '18333.txt', '17000.txt', '4881.txt', '2228.txt', '11471.txt', '3136.txt', '4659.txt', '15617.txt', ' 8063.txt', '7350.txt', '12178.txt', '1721.txt', '13266.txt', '14509.txt', '6728.txt', '1047.txt', '13500. txt', '15171.txt', '8705.txt', '7436.txt', '16478.txt', '11317.txt', '3650.txt', '19993.txt', '10009.txt' , '5221.txt', '18455.txt', '17766.txt', '3888.txt', '6700.txt', '14247.txt', '9433.txt', '13528.txt', ' 12636.txt', '15159.txt', '16450.txt', '4117.txt', '19763.txt', '3678.txt', '17996.txt', '2566.txt', '10021. txt', '5209.txt', '17028.txt', '2200.txt', '10747.txt', '11459.txt', '16336.txt', '4671.txt', '19005.txt' , '7378.txt', '12150.txt', '1709.txt', '6066.txt', '14521.txt ', '9355.txt', '12144.txt', '289.txt', '6072.txt', '9341.txt', '14535.txt', '2214.txt', '10753.txt', '16322.txt', '19011.txt', '4665.txt', '16444.txt', '19777.txt', '4103.txt', '17982.txt', '2572.txt', '10035 .txt', '18469.txt', '6714.txt', '9427.txt']
```


Everything is ready, we can create a data set Dataset, read the file content from the file name list.

```python
import os
class imdbDataset(Dataset):
    def __init__(self,samples_dir):
        self.samples_dir = samples_dir
        self.samples_paths = os.listdir(samples_dir)
    
    def __len__(self):
        return len(self.samples_paths)
    
    def __getitem__(self,index):
        path = self.samples_dir + self.samples_paths[index]
        with open(path,"r",encoding = "utf-8") as f:
            line = f.readline()
            label,tokens = line.split("\t")
            label = torch.tensor([float(label)],dtype = torch.float)
            feature = torch.tensor([int(x) for x in tokens.split(" ")],dtype = torch.long)
            return (feature,label)
    
```

```python
ds_train = imdbDataset(train_samples_path)
ds_test = imdbDataset(test_samples_path)
```

```python
print(len(ds_train))
print(len(ds_test))
```

```
20000
5000
```

```python
dl_train = DataLoader(ds_train,batch_size = BATCH_SIZE,shuffle = True,num_workers=4)
dl_test = DataLoader(ds_test,batch_size = BATCH_SIZE,num_workers=4)

for features,labels in dl_train:
    print(features)
    print(labels)
    break
```

```
tensor([[ 1, 1, 1, ..., 29, 8, 8],
        [13, 11, 247, ..., 0, 0, 8],
        [8587, 555, 12, ..., 3, 0, 8],
        ...,
        [1, 1, 1, ..., 2, 0, 8],
        [618, 62, 25, ..., 20, 204, 8],
        [1, 1, 1, ..., 71, 85, 8]])
tensor([[1.],
        [0.],
        [0.],
        [1.],
        [0.],
        [1.],
        [0.],
        [1.],
        [1.],
        [1.],
        [0.],
        [0.],
        [0.],
        [1.],
        [0.],
        [1.],
        [1.],
        [1.],
        [0.],
        [1.]])
```


Finally, build a model to test whether the data set pipeline is available.

```python
import torch
from torch import nn
import importlib
from torchkeras import Model,summary

class Net(Model):
    
    def __init__(self):
        super(Net, self).__init__()
        
        #After setting the padding_idx parameter, the filled token will always be assigned a 0 vector during the training process
        self.embedding = nn.Embedding(num_embeddings = MAX_WORDS,embedding_dim = 3,padding_idx = 1)
        self.conv = nn.Sequential()
        self.conv.add_module("conv_1",nn.Conv1d(in_channels = 3,out_channels = 16,kernel_size = 5))
        self.conv.add_module("pool_1",nn.MaxPool1d(kernel_size = 2))
        self.conv.add_module("relu_1",nn.ReLU())
        self.conv.add_module("conv_2",nn.Conv1d(in_channels = 16,out_channels = 128,kernel_size = 2))
        self.conv.add_module("pool_2",nn.MaxPool1d(kernel_size = 2))
        self.conv.add_module("relu_2",nn.ReLU())
        
        self.dense = nn.Sequential()
        self.dense.add_module("flatten",nn.Flatten())
        self.dense.add_module("linear",nn.Linear(6144,1))
        self.dense.add_module("sigmoid",nn.Sigmoid())
        
    def forward(self,x):
        x = self.embedding(x).transpose(1,2)
        x = self.conv(x)
        y = self.dense(x)
        return y
        
model = Net()
print(model)

model.summary(input_shape = (200,),input_dtype = torch.LongTensor)

```

```
Net(
  (embedding): Embedding(10000, 3, padding_idx=1)
  (conv): Sequential(
    (conv_1): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
    (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (relu_1): ReLU()
    (conv_2): Conv1d(16, 128, kernel_size=(2,), stride=(1,))
    (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (relu_2): ReLU()
  )
  (dense): Sequential(
    (flatten): Flatten()
    (linear): Linear(in_features=6144, out_features=1, bias=True)
    (sigmoid): Sigmoid()
  )
)
-------------------------------------------------- --------------
        Layer (type) Output Shape Param #
================================================= ==============
         Embedding-1 [-1, 200, 3] 30,000
            Conv1d-2 [-1, 16, 196] 256
         MaxPool1d-3 [-1, 16, 98] 0
              ReLU-4 [-1, 16, 98] 0
            Conv1d-5 [-1, 128, 97] 4,224
         MaxPool1d-6 [-1, 128, 48] 0
              ReLU-7 [-1, 128, 48] 0
           Flatten-8 [-1, 6144] 0
            Linear-9 [-1, 1] 6,145
          Sigmoid-10 [-1, 1] 0
================================================= ==============
Total params: 40,625
Trainable params: 40,625
Non-trainable params: 0
-------------------------------------------------- --------------
Input size (MB): 0.000763
Forward/backward pass size (MB): 0.287796
Params size (MB): 0.154972
Estimated Total Size (MB): 0.443531
-------------------------------------------------- --------------
```

```python
# Compile model
def accuracy(y_pred,y_true):
    y_pred = torch.where(y_pred>0.5,torch.ones_like(y_pred,dtype = torch.float32),
                      torch.zeros_like(y_pred,dtype = torch.float32))
    acc = torch.mean(1-torch.abs(y_true-y_pred))
    return acc

model.compile(loss_func = nn.BCELoss(),optimizer = torch.optim.Adagrad(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy})

```

```python
# Training model
dfhistory = model.fit(10,dl_train,dl_val=dl_test,log_step_freq=200)
```

```
Start Training ...

================================================= ==============================2020-07-11 23:21:53
{'step': 200,'loss': 0.956,'accuracy': 0.521}
{'step': 400,'loss': 0.823,'accuracy': 0.53}
{'step': 600,'loss': 0.774,'accuracy': 0.545}
{'step': 800,'loss': 0.747,'accuracy': 0.56}
{'step': 1000,'loss': 0.726,'accuracy': 0.572}

 +-------+-------+----------+----------+----------- ---+
| epoch | loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+----------- ---+
| 1 | 0.726 | 0.572 | 0.661 | 0.613 |
+-------+-------+----------+----------+----------- ---+

================================================= ==============================2020-07-11 23:22:20
{'step': 200,'loss': 0.605,'accuracy': 0.668}
{'step': 400,'loss': 0.602,'accuracy': 0.674}
{'step': 600,'loss': 0.592,'accuracy': 0.681}
{'step': 800,'loss': 0.584,'accuracy': 0.687}
{'step': 1000,'loss': 0.575,'accuracy': 0.696}

 +-------+-------+----------+----------+----------- ---+
| epoch | loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+----------- ---+
| 2 | 0.575 | 0.696 | 0.553 | 0.716 |
+-------+-------+----------+----------+----------- ---+

================================================= ==============================2020-07-11 23:25:53
{'step': 200,'loss': 0.294,'accuracy': 0.877}
{'step': 400,'loss': 0.299,'accuracy': 0.875}
{'step': 600,'loss': 0.298,'accuracy': 0.875}
{'step': 800,'loss': 0.296,'accuracy': 0.876}
{'step': 1000,'loss': 0.298,'accuracy': 0.875}

 +-------+-------+----------+----------+----------- ---+
| epoch | loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+----------- ---+
| 10 | 0.298 | 0.875 | 0.464 | 0.795 |
+-------+-------+----------+----------+----------- ---+

================================================= ==============================2020-07-11 23:26:19
Finished Training...

```

```python

```

### Three, use DataLoader to load data sets


DataLoader can control the size of the batch, the sampling method of the elements in the batch, and the method of sorting the batch results into the input form required by the model, and can use multiple processes to read data.

The function signature of DataLoader is as follows.

<!-- #region -->
```python
DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    multiprocessing_context=None,
)
```

<!-- #endregion -->

Under normal circumstances, we only configure the five parameters of dataset, batch_size, shuffle, num_workers, drop_last, and use the default values ​​for other parameters.

In addition to the torch.utils.data.Dataset we mentioned earlier, DataLoader can also load another data set torch.utils.data.IterableDataset.

Unlike Dataset, which is equivalent to a list structure, IterableDataset is equivalent to an iterator structure. It is more complex and generally less used.

-dataset: dataset
-batch_size: batch size
-shuffle: Whether out of order
-sampler: sample sampling function, generally no need to set.
-batch_sampler: batch sampling function, generally no need to set.
-num_workers: Use multiple processes to read data and set the number of processes.
-collate_fn: A function to organize a batch of data.
-pin_memory: Whether it is set to lock industry memory. The default is False, the lock industry memory will not use virtual memory (hard disk), and the speed of copying from the lock industry memory to the GPU will be faster.
-drop_last: Whether to drop the last batch of data that the number of samples is less than batch_size.
-timeout: The maximum waiting time for loading a data batch, generally no setting is required.
-worker_init_fn: The initialization function of the dataset in each worker, often used in IterableDataset. Generally not used.



```python
#Build input data pipeline
ds = TensorDataset(torch.arange(1,50))
dl = DataLoader(ds,
                batch_size = 10,
                shuffle = True,
                num_workers=2,
                drop_last = True)
#Iteration data
for batch, in dl:
    print(batch)
```

```
tensor([43, 44, 21, 36, 9, 5, 28, 16, 20, 14])
tensor([23, 49, 35, 38, 2, 34, 45, 18, 15, 40])
tensor([26, 6, 27, 39, 8, 4, 24, 19, 32, 17])
tensor([ 1, 29, 11, 47, 12, 22, 48, 42, 10, 7])
```


**If this book is helpful to you and want to encourage the author, remember to add a star⭐️ to this project and share it with your friends 😊!**

If you need to further communicate with the author on the understanding of the content of this book, please leave a message under the public account "Algorithm Food House". The author has limited time and energy and will respond as appropriate.

You can also reply to keywords in the background of the official account: **Add group**, join the reader exchange group and discuss with you.

![算法美食屋logo.png](./data/算法美食屋二维码.jpg)

```python

```
