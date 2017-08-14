import tensorflow as tf 
import numpy as np

def DenseBlock(input_layer, growth_rate, num_layer, phase,
        bottlenect=True, dropout=0.0, scope='dense_block'):
    with tf.variable_scope(scope):
        input_layers = [input_layer]
        for i in range(num_layer):
            if bottlenect:
                output_layer = BottlenectBlock(input_layers, growth_rate, phase,
                    dropout, scope=('bottlenect_block_%s' % i))
            else:
                output_layer = ConvBlock(input_layers, growth_rate, phase,
                    dropout, scope=('conv_block_%s' % i))
            input_layers.append(output_layer)
    return input_layers

def ConvLayer(x, output_channels, ksize=3, stride=1, name='conv_weight'):
    input_channels = int(x.shape[-1])
    print('[conv %sx%s, input chns:%d, output chns:%d, stride:%d]' 
        % (ksize,ksize, input_channels, output_channels, stride))
    weight = tf.Variable(
    tf.truncated_normal(shape=(ksize, ksize, input_channels, output_channels), stddev=0.001),
            name=name)
    y = tf.nn.conv2d(x, weight, strides=[1,1,1,1], padding='SAME')
    return y

def FcLayer(x, output_size, name='fc'):
    input_size = int(x.shape[-1])
    print('[fc %dx%d]' % (input_size, output_size))
    weight = tf.Variable(
    tf.truncated_normal(shape=(input_size, output_size), stddev=0.001),
        name=name+'_weight')
    bias = tf.Variable(
        tf.truncated_normal(shape=(1, output_size), stddev=0.001),
        name=name+'_bias')
    y = tf.add(tf.matmul(x, weight), bias)
    return y

# BN-CONV1x1-AVGPOOL2x2
def TransitionLayer(input_layers, phase, theta=0.5, dropout=0.0, scope='transition'):
    print('[TransitionLayer]')
    assert(dropout>=0.0 and dropout<1.0)
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.batch_norm(
            tf.concat(input_layers, -1),
            center=True, scale=True, is_training=phase, scope='bn')
        output_channels = int( int(h1.shape[-1]) * theta )
        h2 = ConvLayer(h1, output_channels, 1, 1, 'conv1x1')
        h3 = tf.nn.avg_pool(h2, [1,2,2,1], [1,2,2,1], 'VALID')
        if dropout>0:
            h3 = tf.nn.dropout(h3, 1.0-dropout)
    return h3

# BN-ReLU-CONV3x3
def ConvBlock(input_layers, growth_rate, phase, dropout=0.0, scope='conv_block'):
    print('[ConvBlock]')
    assert(dropout>=0.0 and dropout<1.0)
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.batch_norm(
            tf.concat(input_layers, -1),
            center=True, scale=True, is_training=phase, scope='bn')
        h2 = tf.nn.relu(h1)
        h3 = ConvLayer(h2, growth_rate, 3, 1, 'conv3x3')
        if dropout>0:
            h3 = tf.nn.dropout(h3, 1.0-dropout)
    return h3

# BN-ReLU-CONV1x1-BN-ReLU-CONV3x3
def BottlenectBlock(input_layers, growth_rate, phase, dropout=0.0, scope='bottlenect_bock'):
    print('[BottlenectBlock]')
    assert(dropout>=0.0 and dropout<1.0)
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.batch_norm(
             tf.concat(input_layers, -1),
            center=True, scale=True, is_training=phase, scope='bn1')
        h2 = tf.nn.relu(h1)
        h3 = ConvLayer(h2, 4*growth_rate, 1, 1, 'conv1x1')
        if dropout>0:
            h3 = tf.nn.dropout(h3, 1.0-dropout)
        h4 = tf.contrib.layers.batch_norm(h3, center=True, scale=True,
            is_training=phase, scope='bn2')
        h5 = tf.nn.relu(h4)
        h6 = ConvLayer(h5, growth_rate, 3, 1, 'conv3x3')
        if dropout>0:
            h6 = tf.nn.dropout(h6, 1.0-dropout)
    return h6

def ClassificationBlock(feats, num_class, scope='classification'):
    with tf.variable_scope(scope):
        feat_height = int(feats.shape[1])
        feat_width = int(feats.shape[2])
        channels = int(feats.shape[3])
        avg_score = tf.nn.avg_pool(feats, [1, feat_height ,feat_width, 1], [1,1,1,1], 'VALID')
        x = tf.reshape(avg_score, (-1,channels))
        clf_score = FcLayer(x, num_class, 'clf')
    return clf_score

def ClassificationLoss(predict, target):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict, labels=target))
    return loss

def ClassificationAccuracy(predict, target):
    predict = tf.round(tf.nn.sigmoid(predict))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, target), tf.float32))
    return accuracy

def DenseNetBody(input_layer, growth_rate, phase, block_size, bottlenect, dropout):
    last_layper = input_layer
    for i, size in enumerate(block_size):
        block = DenseBlock(last_layper, growth_rate, size, phase,
            bottlenect, dropout, scope=('dense_block%s' % i))
        trans = TransitionLayer(block, phase, theta=0.5,
            dropout=dropout, scope=('transition%s' %i))
        last_layper = trans
    return last_layper

def DenseNet_CIFAR(imgData, growth_rate, phase, bottlenect=True, dropout=0.0):
    with tf.variable_scope('dense_net_cifar'):
        block0 = ConvLayer(imgData, 32 if bottlenect else 16, 3, 1, 'conv_input')
        # dense block
        dense = DenseNetBody(block0, growth_rate, phase, [6,6,6], bottlenect, dropout)
        # classification
        cls_score = ClassificationBlock(dense, 10)
    return cls_score

# test unit
if __name__ == '__main__':
    sess = tf.Session()
    target = tf.placeholder(tf.float32, (None, 10))
    imgData = tf.placeholder(tf.float32, (None, 32, 32, 3))
    phase = tf.placeholder(tf.bool)
    cls_score = DenseNet_CIFAR(imgData, 6, phase)
    loss = ClassificationLoss(cls_score, target)

    optimizer = tf.train.MomentumOptimizer(0.1, 0.9)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    sess.run(init)

    labels = np.random.randint(0,10,size=(5))
    tmp = np.zeros((5,10))
    tmp[np.arange(5), labels] = 1
    ret = sess.run([train_op, loss], feed_dict={
        imgData:np.random.rand(5, 32, 32, 3), 
        phase:0,
        target:tmp})
    print(ret)