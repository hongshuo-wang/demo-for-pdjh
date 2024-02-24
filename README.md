当前目录下新建三个文件夹，log,savepoint和data
数据集预处理参考process.py文件夹
数据集下载地址
https://nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1TrainPart1.zip
https://nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1TrainPart2.zip
https://nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.0Test.zip
解压后为gnt格式，按代码注释整理到对应的文件夹下执行process.py即可

tensorboard执行命令
tensorboard --logdir=./logs
