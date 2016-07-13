import tensorflow as tf

import tensorflow.contrib.slim as slim
from mnist import inputs, lenet

flags = tf.app.flags
flags.DEFINE_string('train_dir', '/tmp/data',
                    'Directory with the training data.')
flags.DEFINE_integer('batch_size', 5, 'Batch size.')
flags.DEFINE_integer('num_batches', 1000, 'Num of batches to evaluate.')
flags.DEFINE_string('log_dir', './log/eval',
                    'Directory where to log data.')
flags.DEFINE_string('checkpoint_dir', './log/train',
                    'Directory with the model checkpoint data.')
FLAGS = flags.FLAGS


def main(train_dir, batch_size, num_batches, log_dir, checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = log_dir

    images, labels = inputs(train_dir, False, batch_size, num_batches)
    predictions = lenet(images)
    predictions = tf.to_int32(tf.argmax(predictions, 1))

    tf.scalar_summary('accuracy', slim.metrics.accuracy(predictions, labels))

    # These are streaming metrics which compute the "running" metric,
    # e.g running accuracy
    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
        "streaming_mse": slim.metrics.streaming_mean_squared_error(predictions, labels),
    })

    # Define the streaming summaries to write:
    for metric_name, metric_value in metrics_to_values.iteritems():
        tf.scalar_summary(metric_name, metric_value)

    # Evaluate every 30 seconds
    slim.evaluation.evaluation_loop(
        '',
        checkpoint_dir,
        log_dir,
        num_evals=num_batches,
        eval_op=metrics_to_updates.values(),
        summary_op=tf.merge_all_summaries(),
        eval_interval_secs=30)


if __name__=='__main__':
    main(FLAGS.train_dir,
         FLAGS.batch_size,
         FLAGS.num_batches,
         FLAGS.log_dir,
         FLAGS.checkpoint_dir)
