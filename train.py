import os
import numpy as np
import tensorflow as tf
from model import alexnet
from evals import calc_loss_acc, train_op
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_string('f', '', 'kernel')

flags.DEFINE_integer('valid_steps', 11, 'The number of validation steps ')

flags.DEFINE_integer('max_steps', 1001, 'The number of maximum steps for traing')

flags.DEFINE_integer('batch_size', 128, 'The number of images in each batch during training')

flags.DEFINE_float('base_learning_rate', 0.0001, "base learning rate for optimizer")

flags.DEFINE_integer('input_shape', 784, 'The inputs tensor shape')

flags.DEFINE_integer('num_classes', 10, 'The number of label classes')

flags.DEFINE_string('save_dir', './outputs', 'The path to saved checkpoints')

flags.DEFINE_float('keep_prob', 0.75, "the probability of keeping neuron unit")

flags.DEFINE_string('tb_path', './tb_logs/', 'The path points to tensorboard logs ')


FLAGS = flags.FLAGS




def train():
    """Training model

    """
    valid_steps = FLAGS.valid_steps
    max_steps = FLAGS.max_steps
    batch_size = FLAGS.batch_size
    base_learning_rate = FLAGS.base_learning_rate
    input_shape = FLAGS.input_shape  # image shape = 28 * 28
    num_classes = FLAGS.num_classes
    keep_prob = FLAGS.keep_prob
    save_dir = FLAGS.save_dir
    tb_path = FLAGS.tb_path 

    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []

    tf.reset_default_graph()
    # define default tensor graphe 
    with tf.Graph().as_default():
        images_pl = tf.placeholder(tf.float32, shape=[None, input_shape])
        labels_pl = tf.placeholder(tf.float32, shape=[None, num_classes])

        # define a variable global_steps
        global_steps = tf.Variable(0, trainable=False)

        # build a graph that calculate the logits prediction from model
        logits = alexnet(images_pl, num_classes, keep_prob)

        loss, acc, _ = calc_loss_acc(labels_pl, logits)

        # build a graph that trains the model with one batch of example and updates the model params 
        training_op = train_op(loss, global_steps, base_learning_rate)

        # define the model saver
        saver = tf.train.Saver(tf.global_variables())
        
        # define a summary operation 
        summary_op = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # print(sess.run(tf.trainable_variables()))
            # start queue runners
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            train_writter = tf.summary.FileWriter(tb_path, sess.graph)

            # start training
            for step in range(max_steps):
                # get train image / label batch
                train_image_batch, train_label_batch = mnist.train.next_batch(batch_size)

                train_feed_dict = {images_pl: train_image_batch, labels_pl: train_label_batch}

                _, _loss, _acc, _summary_op = sess.run([training_op, loss, acc, summary_op], feed_dict = train_feed_dict)

                # store loss and accuracy value
                train_loss.append(_loss)
                train_acc.append(_acc)
                print("Iteration " + str(step) + ", Mini-batch Loss= " + "{:.6f}".format(_loss) + ", Training Accuracy= " + "{:.5f}".format(_acc))

                if step % 100 == 0:
                    _valid_loss, _valid_acc = [], []
                    print('Start validation process')

                    for step in range(valid_steps):
                        valid_image_batch, valid_label_batch = mnist.test.next_batch(batch_size)

                        valid_feed_dict = {images_pl: valid_image_batch, labels_pl: valid_label_batch}

                        _loss, _acc = sess.run([loss, acc], feed_dict = valid_feed_dict)

                        _valid_loss.append(_loss)
                        _valid_acc.append(_acc)

                    valid_loss.append(np.mean(_valid_loss))
                    valid_acc.append(np.mean(_valid_acc))

                    print("Iteration {}: Train Loss {:6.3f}, Train Acc {:6.3f}, Val Loss {:6.3f}, Val Acc {:6.3f}".format(step, train_loss[-1], train_acc[-1], valid_loss[-1], valid_acc[-1]))
            
            np.save(os.path.join(save_dir, 'accuracy_loss', 'train_loss'), train_loss)
            np.save(os.path.join(save_dir, 'accuracy_loss', 'train_acc'), train_acc)
            np.save(os.path.join(save_dir, 'accuracy_loss', 'valid_loss'), valid_loss)
            np.save(os.path.join(save_dir, 'accuracy_loss', 'valid_acc'), valid_acc)
            checkpoint_path = os.path.join(save_dir, 'model', 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)

