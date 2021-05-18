# Six, Pytorch's high-level API

Pytorch does not have an official high-level API. Generally, nn.Module is used to build a model and write a custom training loop.

In order to train the model more conveniently, the author wrote a Pytorch model interface imitating keras: torchkeras, as a high-level API of Pytorch.

In this chapter, we mainly introduce the following related content of Pytorch's high-level API.

* 3 ways to build a model (inherit the nn.Module base class, use nn.Sequential, and assist in the application of the model container)

* 3 methods of training model (script style, function style, torchkeras.Model class style)

* Use GPU training model (single GPU training, multi-GPU training)


**If this book is helpful to you and want to encourage the author, remember to add a star to this project, and share it with your friends 😊!**

If you need to further communicate with the author on the understanding of the content of this book, please leave a message under the public account "Algorithm Food House". The author has limited time and energy and will respond as appropriate.

You can also reply to the keyword in the background of the official account: **Add group**, join the reader exchange group and discuss with you.

![算法美食屋logo.png](./data/算法美食屋二维码.jpg)
