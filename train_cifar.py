import struct
import numpy as np 
import tensorflow as tf 
import cv2
import time

import densenet 

NUM_CLASS=10
IMG_WIDTH=32
IMG_HEIGHT=32
IMG_CHN=3

def ReadOneImage(fin, label_size=1):
    label = fin.read(label_size)
    if len(label) == 0:
        return None, None
    imgData = fin.read(IMG_HEIGHT*IMG_WIDTH*IMG_CHN)
    label = struct.unpack('B', label)
    imgData = list(struct.iter_unpack('B', imgData))
    imgData = np.array(imgData, np.uint8)
    imgData = np.transpose(imgData.reshape(IMG_CHN, IMG_HEIGHT, IMG_WIDTH), [1,2,0])
    return label, imgData

def LoadDataSet(files):
    imgDatas = []
    labels = []
    for file in files:
        with open(file, 'rb') as fin:
            i = 0
            while True:
                label, imgData = ReadOneImage(fin)
                if label == None:
                    break
                imgDatas.append(imgData)
                labels.append(label)
    return labels, imgDatas

def BatchData(labels, imgDatas, batch_size):
    num_data = len(labels)
    rnd_idxs = np.random.permutation(num_data)
    labels = [labels[idx] for idx in rnd_idxs]
    imgDatas = [imgDatas[idx] for idx in rnd_idxs]
    for i in range(0, num_data, batch_size):
        yield labels[i:i+batch_size],imgDatas[i:i+batch_size]

def PreprocessBatchData(labels, imgDatas):
    batch_size = len(labels)
    batch_labels = np.zeros((batch_size, NUM_CLASS), dtype=np.float32)
    batch_labels[np.arange(batch_size),labels] = 1.0
    batch_imgDatas = tf.concat([DataArguement(img) for img in imgDatas])
    return batch_labels,batch_imgDatas

def DataArguement(imgData):
    imgData = tf.image.random_flip_left_right(imgData)
    imgData = tf.image.random_brightness(imgData)
    imgData = tf.image.random_brightness(imgData, 0.2)
    imgData = tf.image.random_contrast(imgData, 0.8, 1.2)
    pad_imgData = tf.image.resize_image_with_crop_or_pad(imgData, IMG_HEIGHT+8, IMG_WIDTH+8)
    crop_imgData = tf.random_crop(pad_imgData, [IMG_HEIGHT,IMG_WIDTH,IMG_CHN])
    return crop_imgData

def Train(labels, imgDatas, num_epochs=300, batch_size=64):
    sess = tf.Session()

    target = tf.placeholder(tf.float32, (None, NUM_CLASS))
    imgData = tf.placeholder(tf.float32, (None, IMG_HEIGHT, IMG_WIDTH, IMG_CHN))
    phase = tf.placeholder(tf.bool)

    cls_score = densenet.DenseNet_CIFAR(imgData, 12, phase)
    loss = densenet.ClassificationLoss(cls_score, target)

    optimizer = tf.train.MomentumOptimizer(0.1, 0.9)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    sess.run(init)

    for epochs_idx in range(num_epochs):
        print(epochs_idx)
        for batch_labels,batch_imgDatas in BatchData(labels, imgDatas, batch_size):
            batch_labels,batch_imgDatas = PreprocessBatchData(batch_labels, batch_imgDatas)
            feed_dict = {
                imgData:batch_imgDatas,
                target:batch_labels,
                phase:1
            }
            ret = sess.run([train_op,loss], feed_dict)
            print(ret)
            time.sleep(1)


if __name__ == '__main__':
    # binary data files of CIFAR dataset
    files = ['D:/Dataset/cifar-10-batches-bin/data_batch_1.bin',
        'D:/Dataset/cifar-10-batches-bin/data_batch_2.bin',
        'D:/Dataset/cifar-10-batches-bin/data_batch_3.bin',
        'D:/Dataset/cifar-10-batches-bin/data_batch_4.bin',
        'D:/Dataset/cifar-10-batches-bin/data_batch_5.bin']

    labels,imgDatas = LoadDataSet(files)
    print(len(labels))

    Train(labels, imgDatas)