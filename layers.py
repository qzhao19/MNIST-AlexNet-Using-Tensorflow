import tensorflow as tf


def max_pooling(inputs, ksize, stride, padding, name):
    """Create max pooling layer 
    Args:
        inputs: float32 4D tensor
        ksize: a tuple of 2 int with (kernel_height, kernel_width)
        stride: a tuple
        padding: string. padding mode 'SAME', '
        name: string
        
    Returns:
        4D tensor of [batch_size, height, width, channels]
    """

    return tf.nn.max_pool(inputs, ksize=[1, ksize[0], ksize[1], 1], strides=[1, stride[0], stride[1], 1], padding=padding, name=name)

def dropout(inputs, keep_prob, name):
    """Dropout layer

    Args:
        inputs: float32 4D tensor 
        keep_prob: the probability of keep training sample
        name: layer name to define
    Returns:
        4D tensor of [batch_size, height, width, channels]
    """
    return tf.nn.dropout(inputs, rate=(1-keep_prob), name=name)


def norm(inputs, radius=4, name=None):
    """
    
    """
    return tf.nn.lrn(inputs, depth_radius=radius, bias=1.0, alpha=1e-4, beta=0.75, name=name)

# def batch_norm(inputs, name):
#     """batch normalization layer


#     """

def conv2d(inputs, shape, padding, name):
    """Create convolution 2D layer
    Args:
        inputs: float32. 4D tensor
        shape: the shape of kernel 
        padding: string. padding mode 'SAME',
        name: corressponding layer's name
    Returns:
        Output 4D tensor
    """
    with tf.variable_scope(name) as scope:
        # get weights value and record a summary protocol buffer with a histogram
        weights = tf.get_variable('weights', shape=shape, initializer=tf.initializers.he_normal())
        tf.summary.histogram(scope.name + 'weights', weights)

        # get biases value and record a summary protocol buffer with a histogram
        biases = tf.get_variable('biases', shape=shape[3], initializer=tf.initializers.random_normal())
        tf.summary.histogram(scope.name + 'biases', biases)
        # compute convlotion W * X + b, activiation function relu function
        outputs = tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding=padding)
        outputs = tf.nn.bias_add(outputs, biases)
        outputs = tf.nn.relu(outputs, name=scope.name + 'relu')
    return outputs

def fc(inputs, shape, name):
    """Create fully collection layer 
    Args:
        inputs: Float32. 2D tensor with shape [batch, input_units]
        shape: Int. a tuple with [num_inputs, num_outputs]
        name: sring. layer name

    Returns:
        Outputs fully collection tensor
    """

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape = [shape[0], shape[1]])
        biases = tf.get_variable('biases', shape = [shape[1]])
        # outputs = tf.nn.xw_plus_b(inputs, weights, biases, name = scope.name)
        outputs = tf.add(tf.matmul(inputs, weights), biases, name=scope.name)

    return tf.nn.relu(outputs)

