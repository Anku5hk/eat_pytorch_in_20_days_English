# How to eat Pytorch in 20 days ?🔥🔥

<!-- #region -->
**《Eat that pyspark in 10 days》**
* 🚀 github project address: https://github.com/lyhue1991/eat_pyspark_in_10_days
* 🐳 Hewhale column address: https://www.kesci.com/home/column/5fe6aa955e24ed00302304e0 [Code can be run directly in the cloud after fork, no need to configure the environment]


**《Eat that Pytorch in 20 Days》**
* 🚀 github project address: https://github.com/lyhue1991/eat_pytorch_in_20_days
* 🐳 Hewhale column address: https://www.kesci.com/home/column/5f2ac5d8af3980002cb1bc08 [Code can be run directly in the cloud after fork, no need to configure the environment]


** "Eat that TensorFlow 2 in 30 Days" **
* 🚀 github project address: https://github.com/lyhue1991/eat_tensorflow2_in_30_days
* 🐳 Hewhale column address: https://www.kesci.com/home/column/5d8ef3c3037db3002d3aa3a0 [Code can be run directly in the cloud after fork, no need to configure the environment]

### One, Pytorch🔥 or TensorFlow2 🍎

Let me start with the conclusion:

**If you are an engineer, you should choose TensorFlow2 first.**

**If you are a student or researcher, Pytorch should be preferred.**

**If you have enough time, it is best to learn and master TensorFlow2 and Pytorch. **
<!-- #endregion -->

The reasons are as follows:

*1, **The most important thing in the industry is model landing. At present, most domestic Internet companies only support the online deployment of TensorFlow models, not Pytorch. ** And the industry pays more attention to the high availability of the model. Many times the mature model architecture is used, and the need for debugging is not large.


*2, **The most important thing for researchers is to publish articles quickly iteratively, and they need to try some newer model architectures. Pytorch has some advantages over TensorFlow2 in terms of ease of use and is more convenient for debugging. ** And since 2019, it has occupied more than half of the academic world, and there are more corresponding latest research results that can be found.


*3, TensorFlow2 and Pytorch actually have very similar overall styles. After learning one of them, it will be easier to learn the other. If you master both frameworks, you can refer to more open source model cases, and you can easily switch between the two frameworks.


TensorFlow mirroring tutorial of this book:

#### 🍊 "Eat that TensorFlow2 in 30 Days": https://github.com/lyhue1991/eat_tensorflow2_in_30_days

```python

```

### Second, this book is for readers 👼


** This book assumes that the reader has a certain foundation of machine learning and deep learning, and has used Keras or TensorFlow or Pytorch to build and train simple models. **

**For students who do not have any machine learning and deep learning foundations, it is recommended to read the first part of the "Deep Learning Basics" content of the book "Python Deep Learning" when studying this book. **

The book "Python Deep Learning" is written by Francois Chollet, the father of Keras. The book assumes that the reader has no machine learning knowledge and uses Keras as a tool.

Use a wealth of examples to demonstrate the best practices of deep learning. The book is easy to understand. There is no mathematical formula in the whole book, and it focuses on cultivating readers' deep learning intuition. **.

The contents of the 4 chapters of the first part of the book "Python Deep Learning" are as follows, and it is expected that readers will be able to finish it in 20 hours.

* 1. What is deep learning

*2, the mathematical foundation of neural networks

*3, introduction to neural networks

*4, machine learning foundation


```python

```

### Three, the writing style of this book 🍉


**This book is an introductory Pytorch tool that is extremely friendly to human users. Don't let me think is the highest pursuit of this book. **

This book is mainly based on the reference Pytorch official documentation and function doc documentation.

Although the official Pytorch documentation is quite concise and clear, this book has made a lot of optimizations in the chapter structure and selection of examples, which is more user-friendly.

This book is designed in accordance with the degree of difficulty of the content, reader retrieval habits and Pytorch's own hierarchical structure design content, step by step, clear levels, convenient to find corresponding examples according to functions.

This book is as simple and structured as possible in the design of examples to enhance the legibility and versatility of examples. Most of the code snippets are ready to use in practice.

**If the difficulty of mastering Pytorch by learning the official Pytorch documentation is about 5, then the difficulty of learning to master Pytorch through this book should be about 2.**

Only the following figure compares the difference between the official Pytorch document and the book "Eat that Pytorch in 20 Days".



![](./data/Pytorch official vs eat Pytorch.png)

```python

```

### Four, book study plan ⏰

**1, study plan**

This book was written by the author about 3 months after work, and most readers should be able to learn it in 20 days.

It is estimated that the study time spent every day is between 30 minutes and 2 hours.

Of course, this book is also very suitable as a reference for Pytorch's tool manual when the project is implemented.

**Click the blue title of the learning content to enter the chapter. **

| Date | Learning Content | Content Difficulty | Estimated Learning Time | Update Status|
|----:|:------------------------------------------ --------------------|-----------:|----------:|---- -:|
|&nbsp;|[**一、Pytorch's modeling process**](./一、Pytorch's modeling process.md) |⭐️ | 0hour |✅ |
|day1 | [1-1,Structured data modeling process example](./1-1,Structured data modeling process example.md) | ⭐️⭐️⭐️ | 1hour |✅ |
|day2 | [1-2,Picture data modeling process example](./1-2,Picture data modeling process example.md) | ⭐️⭐️⭐️⭐️ | 2hour | ✅ |
|day3 | [1-3,Text data modeling process example](./1-3,Text data modeling process example.md) | ⭐️⭐️⭐️⭐️⭐️ | 2hour | ✅ |
|day4 | [1-4,Time series data modeling process example](./1-4,Time series data modeling process example.md) | ⭐️⭐️⭐️⭐️⭐️ | 2hour | ✅ |
|&nbsp; |[**二. The core concept of Pytorch**](./二. The core concept of Pytorch.md) | ⭐️ | 0hour |✅ |
|day5 | [2-1,Tensor data structure](./2-1,Tensor data structure.md) | ⭐️⭐️⭐️⭐️ | 1hour | ✅ |
|day6 | [2-2,Automatic differentiation mechanism](./2-2,Automatic differentiation mechanism.md) | ⭐️⭐️⭐️ | 1hour | ✅ |
|day7 | [2-3,Dynamic calculation graph](./2-3,Dynamic calculation graph.md) | ⭐️⭐️⭐️⭐️⭐️ | 2hour | ✅ |
|&nbsp; |[**三. Pytorch hierarchy**](./三, Pytorch hierarchy.md) | ⭐️ | 0hour | ✅ |
|day8 | [3-1,Low-level API demonstration](./3-1,Low-level API demonstration.md) | ⭐️⭐️⭐️⭐️ | 1hour | ✅ |
|day9 | [3-2,Intermediate API Demonstration](./3-2,Intermediate API Demonstration.md) | ⭐️⭐️⭐️ | 1hour |✅ |
|day10 | [3-3,High-level API demonstration](./3-3,High-level API demonstration.md) | ⭐️⭐️⭐️ | 1hour |✅ |
|&nbsp; |[**四、Pytorch's low-level API**](./四、Pytorch's low-level API.md) |⭐️ | 0hour| ✅ |
|day11| [4-1, tensor structure operation](./4-1, tensor structure operation.md) | ⭐️⭐️⭐️⭐️⭐️ | 2hour | ✅ |
|day12| [4-2, tensor mathematical operation](./4-2, tensor mathematical operation.md) | ⭐️⭐️⭐️⭐️ | 1hour |✅ |
|day13| [4-3,nn.functional and nn.Module](./4-3,nn.functional and nn.Module.md) | ⭐️⭐️⭐️⭐️ | 1hour |✅ |
|&nbsp; |[**五、Pytorch's Intermediate API**](./五、Pytorch's Intermediate API.md) | ⭐️ | 0hour|✅ |
|day14| [5-1,Dataset and DataLoader](./5-1,Dataset and DataLoader.md) | ⭐️⭐️⭐️⭐️⭐️ | 2hour | ✅ |
|day15| [5-2,model layer](./5-3,model layer.md) | ⭐️⭐️⭐️ | 1hour |✅ |
|day16| [5-3,loss function](./5-4,loss function.md) | ⭐️⭐️⭐️ | 1hour |✅ |
|day17| [5-4,TensorBoardVisualization](./5-4,TensorBoardVisualization.md) | ⭐️⭐️⭐️ | 1hour | ✅ |
|&nbsp; |[**六、Pytorch's high-level API**](./六、Pytorch's high-level API.md)| ⭐️ | 0hour|✅ |
|day18| [6-1,3 ways to build a model](./6-1,3 ways to build a model.md) | ⭐️⭐️⭐️⭐️ | 1hour |✅ |
|day19| [6-2, 3 ways to train a model](./6-2, 3 ways to train a model.md) | ⭐️⭐️⭐️⭐️ | 1hour | ✅ |
|day20| [6-3, Use GPU to train model](./6-3, Use GPU to train model.md) | ⭐️⭐️⭐️⭐️ | 1hour | ✅ |

```python

```

**2, learning environment**

All the source code of this book has been written and tested in jupyter. It is recommended to clone to the local through git and run and learn interactively in jupyter.

In order to be able to open the markdown file directly in jupyter, it is recommended to install jupytext and convert the markdown to an ipynb file.

```python
#Clone the source code of this book to the local, use the code cloud mirror warehouse to download faster in China
#!git clone https://gitee.com/Python_Ai_Road/eat_pytorch_in_20_days

#It is recommended to install jupytext on jupyter notebook so that the markdown files of each chapter of this book can be run as ipynb files
#!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U jupytext
    
#It is recommended to install the latest version of pytorch on jupyter notebook to test the code in this book
#!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U torch torchvision torchtext torchkeras
```

```python
import torch
from torch import nn

print("torch version:", torch.__version__)

a = torch.tensor([[2,1]])
b = torch.tensor([[-1,2]])
c = a@b.t()
print("[[2,1]]@[[-1],[2]] =", c.item())

```

```
torch version: 1.5.0
[[2,1]]@[[-1],[2]] = 0
```

```python

```

### Five, encourage and contact the author 🎈🎈


**If this book is helpful to you and want to encourage the author, remember to add a star to this project, and share it with your friends 😊!**

If you need to further communicate with the author on the understanding of the content of this book, please leave a message under the public account "Algorithm Food House". The author has limited time and energy and will respond as appropriate.

You can also reply to keywords in the background of the official account: **Add group**, join the reader exchange group and discuss with you.

![Algorithm gourmet house logo.png](./data/Algorithm gourmet house QR code.jpg)

```python

```
