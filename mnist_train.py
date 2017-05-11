import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim
from model import lenet
from datasets import mnist

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/data',
                    'Directory with the mnist data.')
flags.DEFINE_integer('batch_size', 5, 'Batch size.')
flags.DEFINE_integer('num_batches', None,
                     'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', './log/train',
                    'Directory with the log data.')
FLAGS = flags.FLAGS


def main(args):
    # load the dataset
    dataset = mnist.get_split('train', FLAGS.data_dir)
    # load the examples in parallel
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    image, label = data_provider.get(['image', 'label'])
    # batch the examples
    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        allow_smaller_final_batch=True)

    # run the image through the model
    predictions = lenet(images)
    # get the cross-entropy loss
    one_hot_labels = slim.one_hot_encoding(
        labels,
        dataset.num_classes)
    slim.losses.softmax_cross_entropy(
        predictions,
        labels)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)

    optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer,
        summarize_gradients=True)

    slim.learning.train(
        train_op,
        FLAGS.log_dir,
        save_summaries_secs=20)


if __name__ == '__main__':
    tf.app.run(argv=sys.argv)
