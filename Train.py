import torch
import Config
from visdom import Visdom       #  python -m visdom.server 启动
if Config.use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
if not Config.use_cuda:
    print("WARNING: It looks like you have a CUDA device, but aren't " +
          "using CUDA.\nRun with --cuda for optimal training speed.")
    torch.set_default_tensor_type('torch.FloatTensor')

import torch.nn as nn
import cv2
import utils
import loss_function
import augmentations
import ssd_net_vgg
import data_myself
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = Config.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets
def xavier(param):
    nn.init.xavier_uniform_(param)
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
def train():
    dataset = data_myself.VOCDetection(
                           transform=augmentations.SSDAugmentation(Config.image_size,
                                                     Config.MEANS))
    data_loader = data.DataLoader(dataset, Config.batch_size,
                                  num_workers=Config.data_load_number_worker,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    net = ssd_net_vgg.SSD()
    vgg_weights = torch.load(Config.pretain_model)

    net.apply(weights_init)
    net.vgg.load_state_dict(vgg_weights)
    # net.apply(weights_init)
    if Config.use_cuda:
        net = torch.nn.DataParallel(net)
        net = net.cuda()
    net.train()
    loss_fun = loss_function.LossFun()
    optimizer = optim.SGD(net.parameters(), lr=Config.lr, momentum=Config.momentum,
                          weight_decay=Config.weight_decacy)
    iter = 0
    step_index = 0
    before_epoch = -1

    viz=Visdom()
    viz.line([50.],[0],win='train_loss',opts=dict(title='train_loss'))
    ii=1
    for epoch in range(1000):
        for step,(img,target) in enumerate(data_loader):
            if Config.use_cuda:
                img = img.cuda()
                target = [ann.cuda() for ann in target]
            img = torch.Tensor(img)
            loc_pre,conf_pre = net(img)
            priors = utils.default_prior_box()
            optimizer.zero_grad()
            loss_l,loss_c = loss_fun((loc_pre,conf_pre),target,priors)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            if iter % Config.print_iter == 0 or before_epoch!=epoch:
                ii+=1
                print('epoch : ',epoch,' iter : ',iter,' step : ',step,' loss : ',loss.item())
                viz.line([loss.item()],[ii],win='train_loss',update='append')
                before_epoch = epoch
            iter+=1
            if iter in Config.lr_steps:
                step_index+=1
                adjust_learning_rate(optimizer,Config.gamma,step_index)
            if iter % Config.save_iter == 0 and iter!=0:
                torch.save(net.state_dict(), 'G:/git_folder/ssd_data_and_weight/data and weight/weights/ssd300_VOC_' +
                           repr(iter) + '.pth')
        if iter >= Config.max_iter:
            break
    torch.save(net.state_dict(), 'G:/git_folder/ssd_data_and_weight/data and weight/weights/ssd_voc_last.pth')

if __name__ == '__main__':
    train()



