# tf-slim-mnist
MNIST tutorial with Tensorflow Slim (tf.contrib.slim) a lightweight library over Tensorflow, you can read more about it [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) and [here](https://github.com/tensorflow/models/blob/master/slim/slim_walkthrough.ipynb) is a good ipython notebook about it

**NOTE** recently updated! tensorflow slim functionality also seems to be moving to other contrib libraries, and if it completely disappears I'll update accordingly


## Setting up data
run `python datasets/download_and_convert_mnist.py` to create [train, validation, test].tfrecords files containing MNIST data
by default (unless you specify `--directory`) they will be put into /tmp/data

## Running
Run the training, validation, and tensorboard concurrently. The results of the training and validation should show up in tensorboard.

### Running the training
run `mnist_train.py` which will read train.tfrecords using an input queue and output its model checkpoints, and summaries to the log directory (you can specify it with `--log_dir`)

### Running the validation
run `mnist_eval.py` which will read validation.tfrecords using an input queue, and also read the train models checkpoints from `log/train` (by default). It will then load the model at that checkpoint and run it on the validation examples, outputting the summaries and log to its own folder `log/eval` (you can specify it with `--log_dir`)

### Running tensorboard
Tensorboard allows you to keep track of your training in a nice and visual way. It will read the logs from the training and validation and should update on its own though you may have to refresh the page manually sometimes.

Make sure both training and validation output their summaries to one log directory and preferably under their own folder. Run `tensorboard --logdir=log` (replace log with your own log folder if you changed it).

If each process has its own folder then train and validation should have their own colour and checkbox
