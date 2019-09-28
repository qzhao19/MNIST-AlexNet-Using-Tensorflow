import tensorflow as f


def calc_loss_acc(labels, logits):
    """Function to compute loss value. Here, we used cross entropy 
    Args:
        logits: 4D tensor. output tensor from segnet model, which is the output of softmax
        labels: true labels tensor
    Returns:
        loss (cross_entropy_mean), accuracy, predicts(logits with softmax) 
    """
    # calc cross entropy mean  cross_entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar(name='loss', tensor=cross_entropy_mean)


    predicts = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))

    accuracy = tf.reduce_mean(tf.cast(predicts, dtype=tf.float32))
    tf.summary.scalar(name='accuracy', tensor=accuracy)

    return cross_entropy_mean, accuracy, predicts


def train_op(total_loss, global_steps, base_learning_rate, option='Adam'):
    """This function defines train optimizer 
    Args:
        total_loss: the loss value
        global_steps: global steps is used to track how many batch had been passed. In the training process, the initial value for global_steps = 0, here  
        global_steps=tf.Variable(0, trainable=False). then after one batch of images passed, the loss is passed into the optimizer to update the weight, then the global 
        step increased by one.
        base_learning_rate: default value 0.1
    Returns:
        the train optimizer
    """

    # base_learning_rate = 0.01
    # get update operation
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):

        if option == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=base_learning_rate)
            print("Running with Adam Optimizer with learning rate:", base_learning_rate)
        elif option == 'SGD':
            # base_learning_rate = 0.01
            learning_rate_decay = tf.train.exponential_decay(base_learning_rate, global_steps, 1000, 0.0005)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate_decay)
            print("Running with SGD Optimizer with learning rate:", learning_rate_decay)
        else:
            raise ValueError('Optimizer is not recognized')

        grads = optimizer.compute_gradients(total_loss, var_list=tf.trainable_variables())
        training_op = optimizer.apply_gradients(grads, global_step=global_steps)

    return training_op



