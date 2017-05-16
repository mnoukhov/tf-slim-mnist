import tensorflow as tf

from datasets import mnist
from model import lenet, load_batch

slim = tf.contrib.slim
metrics = tf.contrib.metrics

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/mnist',
                    'Directory with the MNIST data.')
flags.DEFINE_integer('batch_size', 5, 'Batch size.')
flags.DEFINE_integer('eval_interval_secs', 60,
                    'Number of seconds between evaluations.')
flags.DEFINE_integer('num_evals', 1000, 'Number of batches to evaluate.')
flags.DEFINE_string('log_dir', './log/eval',
                    'Directory where to log evaluation data.')
flags.DEFINE_string('checkpoint_dir', './log/train',
                    'Directory with the model checkpoint data.')
FLAGS = flags.FLAGS


def main(args):
    # load the dataset
    dataset = mnist.get_split('test', FLAGS.data_dir)

    # load batch
    images, labels = load_batch(
        dataset,
        FLAGS.batch_size,
        is_training=False)

    # get the model prediction
    predictions = lenet(images)

    # convert prediction values for each class into single class prediction
    predictions = tf.to_int64(tf.argmax(predictions, 1))

    # streaming metrics to evaluate
    metrics_to_values, metrics_to_updates = metrics.aggregate_metric_map({
        'mse': metrics.streaming_mean_squared_error(predictions, labels),
        'accuracy': metrics.streaming_accuracy(predictions, labels),
    })

    # write the metrics as summaries
    for metric_name, metric_value in metrics_to_values.iteritems():
        tf.summary.scalar(metric_name, metric_value)

    # evaluate on the model saved at the checkpoint directory
    # evaluate every eval_interval_secs
    slim.evaluation.evaluation_loop(
        '',
        FLAGS.checkpoint_dir,
        FLAGS.log_dir,
        num_evals=FLAGS.num_evals,
        eval_op=metrics_to_updates.values(),
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__=='__main__':
    tf.app.run()
