import tensorflow as tf
import numpy as np
import os

'''
'0': 'anger',  # 生气
'1': 'disgust',  # 厌恶
'2': 'fear',  # 恐惧
'3': 'happy',  # 开心
'4': 'sad',  # 伤心
'5': 'surprised',  # 惊讶
'6': 'normal',  # 中性
'''

anger_0_image      = []
anger_0_labels     = []
disgust_1_image    = []
disgust_1_labels   = []
fear_2_image       = []
fear_2_labels      = []
happy_3_image      = []
happy_3_labels     = []
sad_4_image        = []
sad_4_labels       = []
surprised_5_image  = []
surprised_5_labels = []
normal_6_image     = []
normal_6_labels    = []

def get_file(file_dir):
    # 在该路径下遍历0,1,2,3,4,5,6的七个文件夹，将每个文件夹下的所有图片名称放入list中；
    for file in os.listdir(file_dir + '0'):
        anger_0_image.append(file_dir + '0' + '/' + file)
        anger_0_labels.append(0)
    for file in os.listdir(file_dir + '1'):
        anger_0_image.append(file_dir + '1' + '/' + file)
        anger_0_labels.append(1)
    for file in os.listdir(file_dir + '2'):
        anger_0_image.append(file_dir + '2' + '/' + file)
        anger_0_labels.append(2)
    for file in os.listdir(file_dir + '3'):
        anger_0_image.append(file_dir + '3' + '/' + file)
        anger_0_labels.append(3)
    for file in os.listdir(file_dir + '4'):
        anger_0_image.append(file_dir + '4' + '/' + file)
        anger_0_labels.append(4)
    for file in os.listdir(file_dir + '5'):
        anger_0_image.append(file_dir + '5' + '/' + file)
        anger_0_labels.append(5)
    for file in os.listdir(file_dir + '6'):
        anger_0_image.append(file_dir + '6' + '/' + file)
        anger_0_labels.append(6)

    image_list = np.hstack((anger_0_image,disgust_1_image,fear_2_image,happy_3_image,sad_4_image,surprised_5_image,normal_6_image))
    label_list = np.hstack((anger_0_labels,disgust_1_labels,fear_2_labels,happy_3_labels,sad_4_labels,surprised_5_labels,normal_6_labels))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list

def get_batch(image,label,image_W,image_H,batch_size):
    # 改变数据类型
	image1 = tf.cast(image,tf.string)
	label1 = tf.cast(label,tf.int32)

    # 生成一个输出的队列
	input_queue = tf.train.slice_input_producer([image1,label1])
	label = input_queue[1]
    # 通过该队列来读取图像
	image_contents = tf.read_file(input_queue[0])
    # 将图像进行编码处理
	image = tf.image.decode_jpeg(image_contents,channels = 1)

    # 对目标图像进行预处理，改变
	image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    # 对图像进行标准化处理
	image = tf.image.per_image_standardization(image)
    # 生成batch
	image_batch,label_batch = tf.train.batch([image,label],
    											batch_size = batch_size,
    											num_threads = 64,
    											capacity = 256)

	label_batch = tf.reshape(label_batch,[batch_size])
	image_batch = tf.cast(image_batch,tf.float32)

	return image_batch,label_batch