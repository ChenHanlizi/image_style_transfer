# README✨:
如果你觉得这份代码不错，请点点star⭐:吧！  

If you think this code is good, please click star⭐:!
![image](https://github.com/user-attachments/assets/82f318c2-13e7-4417-8e75-73dcb7ad946b)
![image](https://github.com/user-attachments/assets/25cd2525-7dc1-422c-baa4-9319431cf1c9)


<details>
  <summary>中文</summary>

  ## 项目概述

这是一个**基于多尺度特征提取**和**总变差正则化**的图像风格迁移项目。  
为了提升风格迁移图片的视觉质量，提出了一种使用VGG19网络的图像风格迁移方法，通过引入多尺度特征提取和总变差正则化，旨在在生成图像中更全面地保留目标风格的特征并平滑生成图像。首先通过遍历VGG19的层构建了一个包含内容损失和多尺度风格损失的模型；然后使用LBFGS优化器进行迭代优化，同时加入总变差正则化以减少噪声；最后保存生成的图像。
*   运行环境
python3.8，需要GPU
*   文件放置
你需要将风格图片放在Test/style/文件夹下，将原始图片放在Test/content/文件夹下，并修改transfer.py中对应的文件名称。<br>
你可以在这里下载 [VGG19](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)的权重文件。
*   运行代码

    > python transfer.py
</details>

<details>
  <summary>English</summary>

## Project Overview

This is an image style transfer project based on **multi-scale feature extraction** and **total variation regularization**.  
In order to improve the visual quality of style transfer images, an image style transfer method using the VGG19 network is proposed. By introducing multi-scale feature extraction and total variation regularization, it aims to more comprehensively retain the characteristics of the target style in the generated image and smooth the generated image. First, a model with content loss and multi-scale style loss is constructed by traversing the layers of VGG19; then the LBFGS optimizer is used for iterative optimization, and total variation regularization is added to reduce noise; finally, the generated image is saved.
* Running environment
python3.8, GPU required
* File placement
You need to put the style image in the Test/style/ folder, the original image in the Test/content/ folder, and modify the corresponding file name in transfer.py.<br>
You can download the weights file of [VGG19](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth) here.
* Run code

> python transfer.py

</details>
