import tensorflow as tf
from evals import calc_loss_acc, train_op
from layers import max_pooling, dropout, norm, conv2d, fc


def alexnet(inputs, num_classes, keep_prob):
    """Create alexnet model
    """
    x = tf.reshape(inputs, shape=[-1, 28, 28, 1])

    # first conv layer, downsampling layer, and normalization layer
    conv1 = conv2d(x, shape=(11, 11, 1, 96), padding='SAME', name='conv1')
    pool1 = max_pooling(conv1, ksize=(2, 2), stride=(2, 2), padding='SAME', name='pool1')
    norm1 = norm(pool1, radius=4, name='norm1')

    # second conv layer
    conv2 = conv2d(norm1, shape=(5, 5, 96, 256), padding='SAME', name='conv2')
    pool2 = max_pooling(conv2, ksize=(2, 2), stride=(2, 2), padding='SAME', name='pool2')
    norm2 = norm(pool2, radius=4, name='norm2')

    # 3rd conv layer
    conv3 = conv2d(norm2, shape=(3, 3, 256, 384), padding='SAME', name='conv3')
    # pool3 = max_pooling(conv3, ksize=(2, 2), stride=(2, 2), padding='SAME', name='pool3')
    norm3 = norm(conv3, radius=4, name='norm3')

    # 4th conv layer
    conv4 = conv2d(norm3, shape=(3, 3, 384, 384), padding='SAME', name='conv4')

    # 5th conv layer
    conv5 = conv2d(conv4, shape=(3, 3, 384, 256), padding='SAME', name='conv5')
    pool5 = max_pooling(conv5, ksize=(2, 2), stride=(2, 2), padding='SAME', name='pool5')
    norm5 = norm(pool5, radius=4, name='norm5')

    # first fully connected layer
    fc1 = tf.reshape(norm5, shape=(-1, 4*4*256))
    fc1 = fc(fc1, shape=(4*4*256, 4096), name='fc1')
    fc1 = dropout(fc1, keep_prob=keep_prob, name='dropout1')

    fc2 = fc(fc1, shape=(4096, 4096), name='fc2')
    fc2 = dropout(fc2, keep_prob=keep_prob, name='dropout2')

    # output logits value
    with tf.variable_scope('classifier') as scope:
        weights = tf.get_variable('weights', shape=[4096, num_classes], initializer=tf.initializers.he_normal())
        biases = tf.get_variable('biases', shape=[num_classes], initializer=tf.initializers.random_normal())
        # define output logits value
        logits = tf.add(tf.matmul(fc2, weights), biases, name=scope.name + '_logits')

    return logits


