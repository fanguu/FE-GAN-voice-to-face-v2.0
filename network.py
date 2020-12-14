import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchsummary import summary


class VoiceEmbedNet(nn.Module):       # 'channels': [256, 384, 576, 864]
    def __init__(self, input_channel, channels, output_channel=24):
        super(VoiceEmbedNet, self).__init__()
        # 第一层卷积输入特征尺寸： mfcc = 13, fbank = 64, spectrogram = 128
        self.model = nn.Sequential(
            nn.Conv1d(64, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 384, 3, 2, 1, bias=False),
            nn.BatchNorm1d(384, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 576, 3, 2, 1, bias=False),
            nn.BatchNorm1d(576, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(576, 864, 3, 2, 1, bias=False),
            nn.BatchNorm1d(864, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(864, 64, 3, 2, 1, bias=True),

        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(64, 1024),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_channel)
        )

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool1d(x, x.size(2), stride=1)
        x = x.view(x.size(0), -1, 1, 1)     # 按照batchsize的维度将向量B×C×W平铺为一维B×-1×1×1,输入到GAN中
        return x



class Generator(nn.Module):    # 'channels': [1024, 512, 256, 128, 64, 32]
    def __init__(self, input_channel, channels, output_channel, use_attention=0):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(input_channel[1], input_channel[1])
        self.fc = nn.Sequential(
            nn.Linear(input_channel[0] + input_channel[1], 256),
            nn.BatchNorm1d(256, affine=True),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(inplace=True),
        )
        def make_dconv_layer(in_channels, out_channels, kernel_size=4, strides=2, padding=1, bn=False, RL=True, dr=False):
            layers = [nn.ConvTranspose2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=strides, padding=padding, bias=True),]
            if bn:
                layers.append(nn.BatchNorm2d(out_channels, affine=True))   # nn.BatchNorm2d , nn.InstanceNorm2d
            if RL:
                layers.append(nn.LeakyReLU(0.2,inplace=True))  #nn.ReLU(inplace=True)
            if dr:
                layers.append(nn.Dropout(0.25))
            return layers

        # deconvlution networks [1024, 512, 256, 128, 64, 32]
        G_model = [
            *make_dconv_layer(64, channels[0], kernel_size=4, strides=1, padding=0),
            *make_dconv_layer(channels[0], channels[1], kernel_size=4, strides=2, padding=1),
            *make_dconv_layer(channels[1], channels[2], kernel_size=4, strides=2, padding=1),
            *make_dconv_layer(channels[2], channels[3], kernel_size=4, strides=2, padding=1),
            *make_dconv_layer(channels[3], channels[4], kernel_size=4, strides=2, padding=1),
            *make_dconv_layer(channels[4], channels[5], kernel_size=4, strides=2, padding=1),

        ]
        G_model += [nn.ConvTranspose2d(channels[5], output_channel, 1, 1, 0, bias=True), nn.BatchNorm2d(output_channel, affine=True), nn.Tanh()]

        self.model = nn.Sequential(*G_model)

    def forward(self, x):
        # x = torch.cat((self.label_emb(emotion_labels),x),-1)     # 词向量随机生成代替one-hot编码
        # x = self.fc(x)
        x = x.unsqueeze(2).unsqueeze(3)
        x_ = self.model(x)
        return x_

# face embedding network and classify with condition discriminator
class Condititon_D(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(Condititon_D, self).__init__()

        self.label_emb = nn.Embedding(input_channel[1], input_channel[1])

        def make_conv_layer(in_channels, out_channels, kernel_size, stride, padding, bn=True, pool=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                ]
            if bn:
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                # layers.append(nn.Dropout2d(0.25))
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))
            return layers

        # DISCRIMINATOR [32, 64, 128, 256, 512, 512, 64],
        self.model = nn.Sequential(
            *make_conv_layer(input_channel[0]+0, channels[0], kernel_size=1, stride=1, padding=1),   # 返回tuple
            *make_conv_layer(channels[0], channels[1],  4, 2, 1),
            *make_conv_layer(channels[1], channels[2],  4, 2, 1),
            *make_conv_layer(channels[2], channels[3],  4, 2, 1),
            *make_conv_layer(channels[3], channels[4],  4, 2, 1),
            *make_conv_layer(channels[4], channels[5], 4, 2, 1),
            nn.Conv2d(channels[5], channels[6], 4, 2, 0, bias=True),
        )
        # self.fc = nn.Sequential(
        #     # nn.Linear(channels[6], 1024),
        #     # nn.BatchNorm1d(1024),
        #     # nn.LeakyReLU(0.2),
        #     nn.Linear(channels[6], output_channel, bias=False))

    def forward(self, x, emotion_labels):
        # x = torch.cat((emotion_labels, x), 1)
        x = self.model(x)
        # x = x.view(x.size()[0], -1)
        # x = self.fc(x)
        return x

class FaceEmbedNet(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(FaceEmbedNet, self).__init__()

        def make_conv_layer(in_channels, out_channels, kernel_size, stride, padding, RL=True, bn=False, pool=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                ]
            if RL:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                # layers.append(nn.Dropout(0.2))  # DP,
            if bn:
                layers.append(nn.BatchNorm2d(out_channels))
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))
            return layers

        # FaceEmbedNet [32, 64, 128, 256, 512, 512, 64]
        self.model = nn.Sequential(
            *make_conv_layer(input_channel, channels[0], kernel_size=1, stride=1, padding=0),   # 返回tuple
            *make_conv_layer(channels[0], channels[1],  4, 2, 1),
            *make_conv_layer(channels[1], channels[2],  4, 2, 1),
            *make_conv_layer(channels[2], channels[3],  4, 2, 1),
            *make_conv_layer(channels[3], channels[4],  4, 2, 1),
            *make_conv_layer(channels[4], channels[5], 4, 2, 1),
            nn.Conv2d(channels[5], channels[6], 4, 2, 0, bias=True),
        )
        # self.fc = nn.Sequential(
        #     # nn.Linear(channels[6], 1024),
        #     # nn.BatchNorm1d(1024),
        #     # nn.LeakyReLU(0.2),
        #     nn.Linear(channels[6], output_channel, bias=False))

    def forward(self, x):
        x = self.model(x)
        # x = x.view(x.size()[0], -1)
        # x = self.fc(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(Classifier, self).__init__()
        # 线性层
        self.model = nn.Linear(input_channel, output_channel, bias=False)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.model(x)
        return x

def restore_train(g_net,d_net,f1_net,f2_net,c1_net,c2_net):
    g_net.load_state_dict(torch.load('models/generator/01-15,19,18-10000-generator.pth'), strict=True)
    for param in g_net.model.parameters():
        param.requires_grad = False  # model包含的特征层中参数都固定住，不会发生梯度的更新

    d_net.load_state_dict(torch.load('models/14,05-5400-generator.pth'), strict=True)
    for param in d_net.model.parameters():
        param.requires_grad = False  # model包含的特征层中参数都固定住，不会发生梯度的更新

    f1_net.load_state_dict(torch.load('models/14,05-5400-generator.pth'), strict=True)
    for param in d_net.model.parameters():
        param.requires_grad = False  # model包含的特征层中参数都固定住，不会发生梯度的更新

    f2_net.load_state_dict(torch.load('models/14,05-5400-generator.pth'), strict=True)
    for param in d_net.model.parameters():
        param.requires_grad = False  # model包含的特征层中参数都固定住，不会发生梯度的更新

    c1_net.load_state_dict(torch.load('models/14,05-5400-generator.pth'), strict=True)
    for param in d_net.model.parameters():
        param.requires_grad = False  # model包含的特征层中参数都固定住，不会发生梯度的更新

    c2_net.load_state_dict(torch.load('models/14,05-5400-generator.pth'), strict=True)
    for param in d_net.model.parameters():
        param.requires_grad = False  # model包含的特征层中参数都固定住，不会发生梯度的更新


def get_network(net_type, networks_parameters, train=False, test=False):
    net_structure = networks_parameters[net_type]
    net = net_structure['network'](net_structure['input_channel'],
                                   net_structure['channels'],
                                   net_structure['output_channel'])
    optimizer = None

    if networks_parameters['GPU']:
        net.cuda()

    if train and net_type=='g':      #切换到训练模式
        net.train()
        optimizer = optim.Adam(net.parameters(),
                               lr=networks_parameters['lr'],
                               betas=(networks_parameters['beta1'], networks_parameters['beta2']))
    if train and (net_type=='f' or net_type=='d0' or net_type=='c'):
        net.train()
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.003, momentum=0.9)
        optimizer = optim.Adam(net.parameters(),
                               lr=networks_parameters['lr'],
                               betas=(networks_parameters['beta1'], networks_parameters['beta2']))

    if test is True:
        net.eval()
        pretrained_dict = torch.load(net_structure['model_path'])
        model_dict = net.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        optimizer = None

    return net, optimizer

if __name__ == '__main__':
    input_channel = [128, 8]
    channels = [1024, 512, 256, 128, 64,32]   #[1024, 512, 256, 128, 64,32]
    output_channel = 3
    use_attention = 0
    G_net = Generator(input_channel, channels, output_channel, use_attention)
    print(G_net)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G_net = G_net.to(device)
    summary(G_net, (64, 1, 1))

    # input_channel = [3, 8]
    # channels = [32, 64, 128, 256, 512, 512, 64]   #[1024, 512, 256, 128, 64,32]
    # output_channel = 1
    #
    # D_net = Condititon_D(input_channel, channels, output_channel)
    # print(D_net)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # D_net = D_net.to(device)
    # summary(D_net, (11, 128, 128))