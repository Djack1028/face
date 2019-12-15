import input_data as data
train_dir = r'/Volumes/learn/learning-package/project/code_git/data_preprocessing/Training/'

IMG_W = 28
IMG_H = 28
TRAIN_BATCH_SIZE = 32

# 遍历训练集文件夹下的图片，输出图片list和label的list
train,train_labels = data.get_file(train_dir)
# 输出预训练的batch数据
train_batch,train_label_batch = data.get_batch(train,train_labels,IMG_W,IMG_H,TRAIN_BATCH_SIZE)
print(train_label_batch)
print('\n')
print(train_batch)
