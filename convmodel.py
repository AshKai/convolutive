# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h


num_classes = 3
batch_size = 5


# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(i, num_classes)  # [one_hot(float(i), num_classes)]
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        #image = tf.image.rgb_to_grayscale(image, name=None)
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * num_classes, 18 * 33 * 64]), units=5, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=num_classes, activation=tf.nn.softmax)
    return y


example_batch_train, label_batch_train = dataSource(["data/train/0/*.jpg", "data/train/1/*.jpg", "data/train/2/*.jpg"],
                                                    batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["data/valid/0/*.jpg", "data/valid/1/*.jpg", "data/valid/2/*.jpg"],
                                                    batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["data/test/0/*.jpg", "data/test/1/*.jpg", "data/test/2/*.jpg"],
                                                  batch_size=batch_size)

example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train, tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid, tf.float32)))
#cost_test = tf.reduce_sum(tf.square(example_batch_test_predicted - tf.cast(label_batch_test, tf.float32)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    last_valid_data_error = 0
    valid_data_error = 0
    valid_data_listgraf = []
    valid_data_list = []

    for _ in range(200):
        sess.run(optimizer)
        last_valid_data_error = valid_data_error
        valid_data_error = sess.run(cost_valid)
        valid_data_list.append(valid_data_error)
        if _ % 20 == 0:
            print("Iter:", _, "---------------------------------------------")
            print(sess.run(label_batch_valid))
            print(sess.run(example_batch_valid_predicted))

            print "Error:", valid_data_error
            print "Difference:", last_valid_data_error - valid_data_error

    plt.ylabel('Error')
    plt.xlabel('Iteraciones')
    vl_handle, = plt.plot(valid_data_listgraf)
    plt.legend(handles=[vl_handle], labels=['Error validacion'])
    plt.savefig('Grafica_mnist.png')

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
    print

    print("Starting Test...")

    test = sess.run(example_batch_test_predicted)
    y_test = sess.run(label_batch_test)
    prec = 0

    for res_test, real in zip (test, y_test):
        if np.argmax(res_test) == np.argmax(real):
            prec = prec + 1

    test_final = (prec / float(len(y_test)))*100
    print "Precision:", test_final, "%"

    coord.request_stop()
    coord.join(threads)
