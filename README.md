# SSD_pytorch
> 这是github上大神改的SSD代码，把运用到自己项目上所需要的步骤记录下来，顺便做注释
> 原代码的链接[https://github.com/acm5656/ssd_pytorch](https://github.com/acm5656/ssd_pytorch)

1. 首先按照voc0712.py的格式写一个自己的数据读取data_myself.py，再改变一下里面的路径读取方法和类别列表，路径可以归到Config里面

2. 把Train.py里面的参数也归到Config里面

3. Test.py的参数注意调整  注释里标明了

4. 值得注意的问题：1.就算只有一类，在类别列表里也不能只写一个，这是之前代码的bug，偶尔出现，还没有找到原因，所以只做前后景检测的话多加一个类，后果是网络到后面feature map稍微厚点；batch_size不能太小，16以上比较好，学习率也不能太高,从1e-4开始比较适合，不然都容易loss-nan跑飞
