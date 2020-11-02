#!/usr/bin/env python
# coding: utf-8

# keras_gpu_tf2.py
"""
Please note that tensorflow.contrib is totally discarded by the TensorFlow team, the assumed class 
tensorflow.compat.v1.contrib could not be used. So it is necessary to either rewrite all the related 
libraries or include the related modified functions and class(es) into the script. It is much better 
to choose the second way. 

With regard to the update realization, uers can choose the script of Distributed training with Keras. 
https://www.tensorflow.org/tutorials/distribute/keras 

Users also can take the the following lines of code for reference, but could not use the script. 
# -mnist = tf.keras.datasets.mnist
# -(x_train, y_train), (x_test, y_test) = mnist.load_data()

If users have no second GPU, there will be a throwing error as follows. 
Traceback (most recent call last):
  File "keras_gpu_tf2.py", line 393, in <module>
    do_train('cpu')
  File "keras_gpu_tf2.py", line 391, in do_train
    do_train('gpu')
TypeError: 'int' object is not callable
"""


# -import tensorflow as tf
import tensorflow.compat.v1 as tf
# -tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
import numpy as np

import os
import sys
import datetime
import gzip
import collections
# -from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Reshape, Convolution2D, Activation, MaxPooling2D
from keras.optimizers import *
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile


my_dir= os.getenv
print(my_dir)

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
    Args:
        f: A file object that can be passed into a gzip reader.
    Returns:
        data: A 4D uint8 numpy array [index, y, x, depth].
    Raises:
        ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', f.name)

    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)

        return data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].
    Args:
        f: A file object that can be passed into a gzip reader.
        one_hot: Does one hot encoding for the result.
        num_classes: Number of classes for the one hot encoding.
    Returns:
        labels: a 1D uint8 numpy array.
    Raises:
        ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)

    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
          return dense_to_one_hot(labels, num_classes)

        return labels


class DataSet(object):

    def __init__(self, images, labels, fake_data=False, one_hot=False, dtype=dtypes.float32, reshape=True):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:   
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                assert images.shape[3] == 1
                images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
            if dtype == dtypes.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(np.float32)
                images = np.multiply(images, 1.0 / 255.0)
        
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0

            return [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end]


def maybe_download(filename, work_directory, source_url):
    """Download the data from source url, unless it's already here.
    Args:
        filename: string, name of the file in the directory.
        work_directory: string, path to working directory.
        source_url: url to download from if file doesn't exist.
    Returns:
        Path to resulting file.
    """
    if not gfile.Exists(work_directory):
        gfile.MakeDirs(work_directory)

    filepath = os.path.join(work_directory, filename)

    if not gfile.Exists(filepath):
        temp_file_name, _ = urlretrieve_with_retry(source_url)
        gfile.Copy(temp_file_name, filepath)

        with gfile.GFile(filepath) as f:
            size = f.size()

        print('Successfully downloaded', filename, size, 'bytes.')

    return filepath


def read_data_sets(train_dir, fake_data=False, one_hot=False, dtype=dtypes.float32, reshape=True, validation_size=5000):
    if fake_data:
        def fake():
            return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)
        train = fake()
        validation = fake()
        test = fake()

        return Datasets(train=train, validation=validation, test=test)

    TRAIN_IMAGES = '/home/mike/datasets/mnist/train-images-idx3-ubyte.gz'
    TRAIN_LABELS = '/home/mike/datasets/mnist/train-labels-idx1-ubyte.gz'
    TEST_IMAGES = '/home/mike/datasets/mnist/t10k-images-idx3-ubyte.gz'
    TEST_LABELS = '/home/mike/datasets/mnist/t10k-labels-idx1-ubyte.gz'

    local_file = maybe_download(TRAIN_IMAGES, train_dir, SOURCE_URL + TRAIN_IMAGES)
    with open(local_file, 'rb') as f:
        train_images = extract_images(f)

    local_file = maybe_download(TRAIN_LABELS, train_dir, SOURCE_URL + TRAIN_LABELS)
    with open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)

    local_file = maybe_download(TEST_IMAGES, train_dir, SOURCE_URL + TEST_IMAGES)
    with open(local_file, 'rb') as f:
        test_images = extract_images(f)

    local_file = maybe_download(TEST_LABELS, train_dir, SOURCE_URL + TEST_LABELS)
    with open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    validation = DataSet(validation_images, validation_labels, dtype=dtype, reshape=reshape)
    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

    return Datasets(train=train, validation=validation, test=test)


def load_mnist(train_dir='MNIST-data'):
# -def load_mnist(train_dir='/home/mic/datasets/mnist'):
    return read_data_sets(train_dir)


learning_rate = 0.001
training_epochs = 2
batch_size = 100
display_step = 1

mnist = read_data_sets('/home/mike/datasets/mnist', one_hot=True)
trainimg    = mnist.train.images
trainlabel  = mnist.train.labels
testimg     = mnist.test.images
testlabel   = mnist.test.labels


def do_train(device):
    if device == 'gpu': # Train with the first GPU
        # -device_type = '/gpu:0'
        device_type = 'gpu:0' 
    else:
        # device_type = '/cpu:0'
        device_type = 'cpu:0' # Traun with the first CPU
        
    with tf.device(device_type): # <= This is optional
        n_input  = 784
        n_output = 10
        weights  = {
            'wc1': tf.Variable(tf.random.normal([3, 3, 1, 64], stddev=0.1)),
            'wd1': tf.Variable(tf.random.normal([14*14*64, n_output], stddev=0.1))
        }
        biases   = {
            'bc1': tf.Variable(tf.random.normal([64], stddev=0.1)),
            'bd1': tf.Variable(tf.random.normal([n_output], stddev=0.1))
        }

        def conv_simple(_input, _w, _b):
            # Reshape input
            _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
            # Convolution
            _conv1 = tf.nn.conv2d(input=_input_r, filters=_w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
            # Add-bias
            _conv2 = tf.nn.bias_add(_conv1, _b['bc1'])
            # Pass ReLu
            _conv3 = tf.nn.relu(_conv2)
            # Max-pooling
            _pool  = tf.nn.max_pool2d(input=_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Vectorize
            _dense = tf.reshape(_pool, [-1, _w['wd1'].get_shape().as_list()[0]])
            # Fully-connected layer
            _out = tf.add(tf.matmul(_dense, _w['wd1']), _b['bd1'])
            # Return everything
            out = {
                'input_r': _input_r, 'conv1': _conv1, 'conv2': _conv2, 'conv3': _conv3
                , 'pool': _pool, 'dense': _dense, 'out': _out
            }

            return out

        def conv_keras(_input):
            # Reshape input
            _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
            # Convolution2D(nb_filters, kernal_size[0], kernal_size[1])
            _conv1 = Convolution2D(64,3,3, padding='same', input_shape=(28,28,1))(_input_r)
            _relu1 = Activation('relu')(_conv1)
            _pool1 = MaxPooling2D(pool_size=(2,2))(_relu1)
            # Conv layer 2
            _conv2 = Convolution2D(64,3,3, padding='same')(_pool1)
            _relu2 = Activation('relu')(_conv2)
            _pool2 = MaxPooling2D(pool_size=(2,2))(_relu2)
            # FC layer 1
            _dense1 = tf.reshape(_pool2, [-1, np.prod(_pool2.get_shape()[1:].as_list())])
            _dense2 = Dense(128, activation='relu')(_dense1)
            preds = Dense(10, activation='softmax')(_dense2)

            return preds

    print ("CNN ready with {}".format(device_type))

    x = tf.compat.v1.placeholder(tf.float32, [None, n_input])
    y = tf.compat.v1.placeholder(tf.float32, [None, n_output])
   
    with tf.device(device_type):
        _pred = conv_keras(x)
        cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(y), logits=_pred))
        optm = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        _corr = tf.equal(tf.argmax(input=_pred,axis=1), tf.argmax(input=y,axis=1)) # Count corrects
        accr = tf.reduce_mean(input_tensor=tf.cast(_corr, tf.float32)) # Accuracy
        init = tf.compat.v1.global_variables_initializer()

    print ("Network Ready to Go!")
    
    do_train = 1 # Train with the second device (GPU or CPU)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    
    start_time = datetime.datetime.now()

    if do_train == 1: # Train with the first device (GPU or CPU)
        for epoch in range(training_epochs):
            avg_cost = 0.
            # -total_batch = int(mnist.train.num_examples/batch_size)
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Fit training using batch data
                sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch

            # Display logs per epoch step
            if epoch % display_step == 0: 
                print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
                train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})
                print (" Training accuracy: %.3f" % (train_acc))
                test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel})
                print (" Test accuracy: %.3f" % (test_acc))

            # Save Net
              # -if epoch % save_step == 0:
                    # -saver.save(sess, "nets/cnn_mnist_simple.ckpt-" + str(epoch))
        print ("Optimization Finished.")
        print ("Single {} computaion time : {}".format(device, datetime.datetime.now() - start_time))

        do_train('gpu')
        
do_train('cpu')