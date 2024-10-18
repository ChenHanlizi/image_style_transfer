#style多尺度
# torch import
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# other import
import numpy as np
from PIL import Image
import copy

# VGG19权重文件路径
vgg19_weights_path = 'vgg19-dcbb9e9d.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图像路径
path_style = 'Test/style/blue.jpg'
path_neirong = 'Test/content/lena.jpg'

output_folder = 'output2/'
os.makedirs(output_folder, exist_ok=True)

# 权重
style_weight = 1000
content_weight = 1

# 图像大小
image_size = 256


# pipeline
pipeline = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# 图片加载函数
# 输入图片路径，输出与网络匹配的张量
def image_loader(img_path):
    img = Image.open(img_path)
    img_tensor = pipeline(img)
    # 在第一维添加一个维度，为输入网络需要
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


# 在GPU上运算
style_img = image_loader(path_style).to(device)
content_img = image_loader(path_neirong).to(device)

assert style_img.size() == content_img.size(), "两张图象大小需要相等"

unload = transforms.ToPILImage()


# 加载vgg19网络
cnn = models.vgg19(pretrained=False)
cnn.load_state_dict(torch.load(vgg19_weights_path))
cnn.features.to(device)
#cnn = models.vgg19(pretrained=True).features
#cnn.to(device)
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


# 内容损失，它是一层网络，为nn.module的子类
class ContentLoss(nn.Module):
    # target是内容输入网络的结果
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # detach()可以将target这几层特征图与之前的动态图解耦，这样就不会操作到原来的特征图
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    # 用以计算目标与输入的误差
    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    # retain_graph 如果设置为False，计算图中的中间变量在计算完后就会被释放
    # 进行一次backward之后，各个节点的值会清除，这样进行第二次backward会报错，如果加上retain_graph==True后,可以再来一次backward。
    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


# 获得gram矩阵函数
def Gram(input):
    a, b, c, d = input.size()
    # 将特征图展平为单一向量
    features = input.view(a * b, c * d)
    # feature与其转置相乘，相当于任意两数相乘
    G = torch.mm(features, features.t())
    # 归一化
    return G.div(a * b * c * d)


# 风格损失
class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        input = input.cuda()

        # 获取多尺度特征
        # scales = [0.5, 0.75, 1.0, 1.25, 1.5]  # 可根据需要调整
        scales = [1.0, 1.25, 1.5]  # 可根据需要调整
        style_loss = 0

        for scale in scales:
            scaled_input = F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=False)
            G = Gram(scaled_input)
            G.mul_(self.weight)
            style_loss += self.criterion(G, self.target)

        self.loss = style_loss / len(scales)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss



model = nn.Sequential()
model.to(device)
content_losses = []
style_losses = []

# 构建 model
i = 1
#for layer in list(cnn):
for layer in cnn.features:
    # 获得卷积层
    if isinstance(layer, nn.Conv2d):
        name = 'conv_' + str(i)
        model.add_module(name, layer)

        if name in content_layers:
            # 把内容图像传入模型，获取需要达到的特征图
            target = model(content_img).clone()
            # 实例化content_loss层，和其他如conv2d层相似
            content_loss = ContentLoss(target, content_weight)
            content_loss = content_loss.to(device)
            model.add_module('content_loss_' + str(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).clone()
            target_feature = target_feature.to(device)
            target_feature_gram = Gram(target_feature)
            style_loss = StyleLoss(target_feature_gram, style_weight)
            style_loss = style_loss.to(device)
            model.add_module('style_loss_' + str(i), style_loss)
            style_losses.append(style_loss)

    if isinstance(layer, nn.ReLU):
        name = 'relu_' + str(i)
        model.add_module(name, layer)
        i += 1
    if isinstance(layer, nn.MaxPool2d):
        name = 'pool_' + str(i)
        model.add_module(name, layer)


input_img = torch.randn(content_img.size()).to(device)

# ... （之前的代码）

# 迭代开始
# nn.Parameter将张量转换为可以反向传播的对象
input_parm = nn.Parameter(input_img.data)
# 仅将输入图像传入优化器，仅对输入图像进行反向传播
optimizer = optim.LBFGS([input_parm], lr=1)
num_step = 700
output_interval = 100  # 设置每100轮输出一次图片

print('正在构造风格迁移模型')
print('开始优化')

for i in range(num_step):
    input_parm.data.clamp_(0, 1)
    optimizer.zero_grad()
    # 这一步会运行 forward
    model(input_parm)
    style_score = 0
    content_score = 0

    for sl in style_losses:
        style_score += sl.backward()
    for cl in content_losses:
        content_score += cl.backward()

    if i % 50 == 0:
        print('正在运行{}轮'.format(i))
        print('风格损失{},\t内容损失{}'.format(style_score, content_score))

    if i % output_interval == 0:
        output_path = os.path.join(output_folder, f'output_{i}.png')
        # 保存生成的图片
        output_img = input_parm.data.clamp_(0, 1).cpu().squeeze(0)
        output_img = transforms.ToPILImage()(output_img)
        output_img.save(output_path)

    def closure():
        return style_score + content_score

    optimizer.step(closure)

# 最后一轮也保存一次图片
output_path = os.path.join(output_folder, f'output_{num_step}.png')
output_img = input_parm.data.clamp_(0, 1).cpu().squeeze(0)
output_img = transforms.ToPILImage()(output_img)
output_img.save(output_path)
