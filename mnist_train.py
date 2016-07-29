import tensorflow as tf
import tensorflow.contrib.slim as slim
from mnist import inputs, lenet

flags = tf.app.flags
flags.DEFINE_string('train_dir', '/tmp/data',
                    'Directory with the training data.')
flags.DEFINE_integer('batch_size', 5, 'Batch size.')
flags.DEFINE_integer('num_batches', None, 'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', './log/train',
                    'Directory with the log data.')
FLAGS = flags.FLAGS


def main(train_dir, batch_size, num_batches, log_dir):
    images, labels = inputs(train_dir,
                            True,
                            batch_size,
                            num_batches,
                            one_hot_labels=True)
    predictions = lenet(images)

    slim.losses.softmax_cross_entropy(predictions, labels)
    total_loss = slim.losses.get_total_loss()
    tf.scalar_summary('loss', total_loss)

    optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
    train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)

    slim.learning.train(train_op, log_dir, save_summaries_secs=20)


if __name__ == '__main__':
    main(FLAGS.train_dir,
         FLAGS.batch_size,
         FLAGS.num_batches,
         FLAGS.log_dir)
