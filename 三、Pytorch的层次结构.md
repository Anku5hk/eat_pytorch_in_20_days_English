# Three, the hierarchy of Pytorch


In this chapter, we introduce 5 different hierarchical structures in Pytorch: hardware layer, kernel layer, low-level API, mid-level API, and high-level API [torchkeras]. And take linear regression and DNN binary classification models as examples to visually compare and show the characteristics of models implemented at different levels.

The hierarchical structure of Pytorch can be divided into the following five layers from low to high.

The bottom layer is the hardware layer. Pytorch supports CPU and GPU to join the computing resource pool.

The second layer is the core implemented in C++.

The third layer is the operators implemented in Python, providing low-level API instructions that encapsulate the C++ kernel, mainly including various tensor operation operators, automatic differentiation, and variable management.
Such as torch.tensor, torch.cat, torch.autograd.grad, nn.Module.
If the model is compared to a house, then the third layer of API is [model brick].

The fourth layer is the model component implemented in Python, which encapsulates the low-level API, mainly including various model layers, loss functions, optimizers, data pipelines, and so on.
Such as torch.nn.Linear, torch.nn.BCE, torch.optim.Adam, torch.utils.data.DataLoader.
If the model is compared to a house, then the fourth layer of API is [Wall of Model].

The fifth layer is the model interface implemented by Python. Pytorch does not have an official high-level API. In order to facilitate the training of the model, the author imitated the model interface in keras and used less than 300 lines of code to encapsulate the high-order model interface torchkeras.Model of Pytorch. If the model is compared to a house, then the fifth layer of API is the model itself, that is, the [model house].


**If this book is helpful to you and want to encourage the author, remember to add a star⭐️ to this project and share it with your friends 😊!**

If you need to further communicate with the author on the understanding of the content of this book, please leave a message under the public account "Algorithm Food House". The author has limited time and energy and will respond as appropriate.

You can also reply to the keyword in the background of the official account: **Add group**, join the reader exchange group and discuss with you.

![算法美食屋logo.png](./data/算法美食屋二维码.jpg)
