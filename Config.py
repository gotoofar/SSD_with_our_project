import os.path as osp
sk = [ 15, 30, 60, 111, 162, 213, 264 ]    #框的大小
feature_map = [ 38, 19, 10, 5, 3, 1 ]
steps = [ 8, 16, 32, 64, 100, 300 ]    #需要遍历的长度和宽度 300/[ 8, 16, 32, 64, 100, 300 ]  最深的那个图 只需要在最中间找到 264*264的框
image_size = 300
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]   #预设anchors的长宽比
MEANS = (104, 117, 123)
batch_size = 16
data_load_number_worker = 0
lr = 1e-4
momentum = 0.9
weight_decacy = 5e-4
gamma = 0.1
use_cuda = True
lr_steps = (200, 400, 800)   #改变学习率的代数
max_iter = 800    #最大训练代数
class_num = 3

pretain_model='G:/git_folder/ssd_pytorch-master/ssd_pytorch-master/weights/vgg16_reducedfc.pth'   #预训练模型位置
print_iter=5     #多少代打印一次信息
save_iter=200   #每多少代保存一次参数
ImageSets='G:/git_folder/ssd_pytorch-master/ssd_pytorch-master/VOC_stone/ImageSet/Main/train.txt'   #imageset的txt位置
rootpath='G:/git_folder/ssd_pytorch-master/ssd_pytorch-master/VOC_stone/'   #存放图片和标注文件的位置
Annotation_name='Annotations'   #标注文件夹名
Jpegimage_name='JPEGImages'    #图片文件夹名
