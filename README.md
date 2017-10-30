# tf-slim-mnist

MNIST tutorial with Tensorflow Slim (tf.contrib.slim) a lightweight library over Tensorflow, you can read more about it [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim). 
[Here](https://github.com/tensorflow/models/blob/master/research/slim/slim_walkthrough.ipynb) is a good iPython notebook about it.

## Setting up data

Run `python datasets/download_and_convert_mnist.py` to create {train, test}.tfrecords files containing MNIST data
by default (unless you specify `--directory`) they will be put into /tmp/mnist

## Running

Run the training, validation, and tensorboard concurrently. The results of the training and validation should show up in tensorboard.

### Running the training

run `mnist_train.py` which will read train.tfrecords using an input queue and output its model checkpoints, and summaries to the log directory (you can specify it with `--log_dir`)

### Running the validation

Run `mnist_eval.py` which will read test.tfrecords using an input queue, and also read the train models checkpoints from `log/train` (by default). It will then load the model at that checkpoint and run it on the testing examples, outputting the summaries and log to its own folder `log/eval` (you can specify it with `--log_dir`)

### Running TensorBoard

TensorBoard allows you to keep track of your training in a nice and visual way. It will read the logs from the training and validation and should update on its own though you may have to refresh the page manually sometimes.

Make sure both training and validation output their summaries to one log directory and preferably under their own folder. Run `tensorboard --logdir=log` (replace log with your own log folder if you changed it).

If each process has its own folder then train and validation should have their own colour and checkbox

## Notes

### Woah, data input seems pretty different from what it used to be

TensorFlow has really changed the way they're doing data input (for the better!) and though the new way seems pretty complicated (with queue runners etc...) it isn't that bad and can potentially make everything much {faster,better}.

I'm trying to keep up with all the changes but if something seems off to you, then please open an issue or create a pull request!

### Where did you get all those files in `/dataset` ?

I took those files from the [`tensorflow/models`](https://github.com/tensorflow/models/) repo in the TensorFlow `slim` folder [here](https://github.com/tensorflow/models/tree/master/research/slim). I modified `download_and_convert_mnist.py` just a little so it can be run as a standalone program, and took only the files you need to run a LeNet architecture for the mnist dataset.

### How do I do more than MNIST?

Modify the model file with whatever model you want, change the data input (maybe look at the [datasets already available in slim](https://github.com/tensorflow/models/tree/master/research/slim/datasets)).

