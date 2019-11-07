# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import l2norm
import torch.nn.functional as F
import Config
num_class=Config.num_class

feature_num1=256
feature_num2=1024
feature_num3=256
feature_num4=256
feature_num5=256
feature_num6=256

class ResidualBlock(nn.Module):
    '''
    实现子module: Residual Block
    '''

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet,self).__init__()
        #
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),      # 150
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)     #75
        )

        #
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, feature_num1, 4, stride=2)    #38
        self.layer3 = self._make_layer(feature_num1, 512, 6, stride=2)   #19
        self.layer4 = self._make_layer(512, feature_num2, 3, stride=1)   #



        #
        #self.fc = nn.Linear(512, num_classes)

        self.conv8_1 = nn.Sequential(
            nn.Conv2d(in_channels=feature_num2,out_channels=256,kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv8_2 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=feature_num3,kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv9_1 = nn.Sequential(
            nn.Conv2d(in_channels=feature_num3, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv9_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=feature_num4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv10_1 = nn.Sequential(
            nn.Conv2d(in_channels=feature_num4, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv10_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=feature_num5, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv11_1 = nn.Sequential(
            nn.Conv2d(in_channels=feature_num5, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv11_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=feature_num6, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        # 特征层位置输出
        self.feature_map_loc_1 = nn.Sequential(
            nn.Conv2d(in_channels=feature_num1, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_loc_2 = nn.Sequential(
            nn.Conv2d(in_channels=feature_num2, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_loc_3 = nn.Sequential(
            nn.Conv2d(in_channels=feature_num3, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_loc_4 = nn.Sequential(
            nn.Conv2d(in_channels=feature_num4, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_loc_5 = nn.Sequential(
            nn.Conv2d(in_channels=feature_num5, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_loc_6 = nn.Sequential(
            nn.Conv2d(in_channels=feature_num6, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
        )
        # 特征层类别输出
        self.feature_map_conf_1 = nn.Sequential(
            nn.Conv2d(in_channels=feature_num1, out_channels=4 * num_class, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_conf_2 = nn.Sequential(
            nn.Conv2d(in_channels=feature_num2, out_channels=6 * num_class, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_conf_3 = nn.Sequential(
            nn.Conv2d(in_channels=feature_num3, out_channels=6 * num_class, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_conf_4 = nn.Sequential(
            nn.Conv2d(in_channels=feature_num4, out_channels=6 * num_class, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_conf_5 = nn.Sequential(
            nn.Conv2d(in_channels=feature_num5, out_channels=4 * num_class, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_conf_6 = nn.Sequential(
            nn.Conv2d(in_channels=feature_num6, out_channels=4 * num_class, kernel_size=3, stride=1, padding=1)
        )

    def _make_layer(self, inchannel, outchannel, bloch_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, bloch_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)








    def forward(self, image):
        out = self.pre(image)

        out = self.layer1(out)
        out = self.layer2(out)   #38

        my_L2Norm = l2norm.L2Norm(feature_num1, 20)
        feature_map_1 = out
        feature_map_1 = my_L2Norm(feature_map_1)
        loc_1 = self.feature_map_loc_1(feature_map_1).permute((0, 2, 3, 1)).contiguous()
        conf_1 = self.feature_map_conf_1(feature_map_1).permute((0, 2, 3, 1)).contiguous()

        out = self.layer3(out)    #19

        out = self.layer4(out)   #19

        feature_map_2 = out
        loc_2 = self.feature_map_loc_2(feature_map_2).permute((0, 2, 3, 1)).contiguous()
        conf_2 = self.feature_map_conf_2(feature_map_2).permute((0, 2, 3, 1)).contiguous()
        out = self.conv8_1(out)
        out = self.conv8_2(out)

        feature_map_3=out
        loc_3 = self.feature_map_loc_3(feature_map_3).permute((0, 2, 3, 1)).contiguous()
        conf_3 = self.feature_map_conf_3(feature_map_3).permute((0, 2, 3, 1)).contiguous()
        out = self.conv9_1(out)
        out = self.conv9_2(out)    #5    256

        feature_map_4 = out
        loc_4 = self.feature_map_loc_4(feature_map_4).permute((0, 2, 3, 1)).contiguous()
        conf_4 = self.feature_map_conf_4(feature_map_4).permute((0, 2, 3, 1)).contiguous()
        out = self.conv10_1(out)
        out = self.conv10_2(out)  #3


        feature_map_5 = out
        loc_5 = self.feature_map_loc_5(feature_map_5).permute((0, 2, 3, 1)).contiguous()
        conf_5 = self.feature_map_conf_5(feature_map_5).permute((0, 2, 3, 1)).contiguous()
        out = self.conv11_1(out)
        out = self.conv11_2(out)  #1


        feature_map_6 = out
        loc_6 = self.feature_map_loc_6(feature_map_6).permute((0, 2, 3, 1)).contiguous()
        conf_6 = self.feature_map_conf_6(feature_map_6).permute((0, 2, 3, 1)).contiguous()

        loc_list = [loc_1, loc_2, loc_3, loc_4, loc_5, loc_6]
        conf_list = [conf_1, conf_2, conf_3, conf_4, conf_5, conf_6]
        return loc_list, conf_list




























