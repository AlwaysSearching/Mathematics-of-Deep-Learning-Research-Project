import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.image import random_crop, resize_with_crop_or_pad
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy

import time
import numpy as np
import pickle as pkl

from models.conv_nets import make_convNet
from models.resnet import make_resnet18_UniformHe


def train_conv_nets(
    data_set,
    convnet_depth,
    convnet_widths,
    label_noise_as_int=10,
    augment_data=False,
    scaled_loss_alpha=None,
    layer_initializer=None,
    n_batch_steps=500_000,
    optimizer=None,
    save=True,
    data_save_path_prefix="",
    data_save_path_suffix="",
    load_saved_metrics=False
):
    """
    Train and save the results of Conv nets of a given range of model widths.

    Note: 500_000 is approximately 1250 epochs. 1_600_000 batchs is approximately 4K epochs.

    Parameters
    ----------

    data_set: str
        Which data set to train on. See the load data funciton.
    convnet_depth: int
    convnet_widths: list[int]
        List of model widths to train.
    label_noise_as_int: int
        Percentage of label noise to add to the training data.
    augment_data: bool
        Whether of not to apply random cropping and random left right flipping to the training data. 
        Default value is set to false. Augmentation is applied  at train time.
    scaled_loss_alpha: float or str
        The alplha value used to scale the cross entropy loss used during training. If '1_m' or '1_sqrt_m' is passed
        then alpha will be a function of the width parameter.
    layer_initializer: tf.keras.initializers
        Pass a specific initialization method to use in initializing model weights. default is He uniform.
    n_batch_steps: int
        number of gradient descent steps to take.
    save: bool
        whether to save the data and trained model weights.
    data_save_path_prefix: str
        prefix to add to the save pkl file path.
    data_save_path_suffix: str
        suffix to add to the save pkl file name.
    load_saved_metrics: bool
        if True, will attempt to load the metrics from a previous training session in the save_path,
        to continue training from there. If True, will load the saved .pkl file instead of starting
        over and overwriting it. 
    """

    label_noise = label_noise_as_int / 100

    # load the relevent dataset. Note that the training data is cast to tf.float32 and normalized by 255.
    (x_train, y_train), (x_test, y_test), image_shape = load_data(data_set, label_noise)

    batch_size = 128
    # total number desirec SGD steps / number batches per epoch = n_epochs
    n_epochs = n_batch_steps // (x_train.shape[0] // batch_size)
    n_classes = tf.math.reduce_max(y_train).numpy() + 1
    
    # store results for later graphing and analysis.
    model_histories = {}
    metrics = {}

    # Paths to save model weights and
    model_weights_paths = f"trained_model_weights_{data_set}/conv_nets_depth_{convnet_depth}_{label_noise_as_int}pct_noise_alpha_{alpha}/"
    data_save_path = (
        "experimental_results_{}/conv_nets_depth_{}_{}pct_noise_alpha_{}".format(
            data_set, convnet_depth, label_noise_as_int, alpha
        ).replace(".", "_")
        + ".pkl"
    )

    # add possilbe data save path identifiers.
    if data_save_path_prefix:
        model_weights_paths = data_save_path_prefix + '/' + model_weights_paths
        data_save_path = data_save_path_prefix + '/' + data_save_path

    if data_save_path_suffix:
        assert data_save_path[-4:] == ".pkl"
        data_save_path = data_save_path[:-4] + data_save_path_suffix + ".pkl"
    
    # load data from prior runs of related experiment.
    if load_saved_metrics:
        try:
            with open(data_save_path, 'rb') as f:
                metrics = pkl.load(f)
        except Exception as e:
            print('Could not find saved metrics.pkl file, exiting')
            raise e

        loaded_widths = [int(i.split('_')[-1]) for i in metrics.keys()]
        assert convnet_widths[:len(loaded_widths)] == loaded_widths
        print('loaded results for width %s from existing file at %s' %(', '.join([str(i) for i in loaded_widths]), data_save_path))

        assert data_save_path[-4:] == ".pkl"
        data_backup_path = data_save_path[:-4] + 'backup_w%d_' %loaded_widths[-1] + time.strftime("%D_%H%M%S").replace('/', '') + ".pkl"
        print('saving existing result.pkl to backup at %s' %data_backup_path)
        pkl.dump(metrics, open(data_backup_path, "wb"))

    for width in convnet_widths:
        if scaled_loss_alpha is None:
            # # some early experiments might have not specified from_logits=True
            # scaled_loss = "sparse_categorical_crossentropy"
            scaled_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        elif scaled_loss_alpha=='1_m' or scaled_loss_alpha=='1_sqrtm':
            scaled_loss = get_scaled_sparse_categorical_loss(scaled_loss_alpha, width)
        else:
            scaled_loss = get_scaled_sparse_categorical_loss(scaled_loss_alpha)

        if load_saved_metrics and width in loaded_widths:
            print('width %d results already loaded from .pkl file, training skipped' %width)
            continue

        # Depth 5 Conv Net using default Kaiming Uniform Initialization.
        conv_net, model_id = make_convNet(
            image_shape, depth=convnet_depth, init_channels=width, n_classes=n_classes, layer_initializer=layer_initializer
        )

        conv_net.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=inverse_squareroot_lr())
            if optimizer is None
            else optimizer,
            loss=scaled_loss,
            metrics=["accuracy"],
        )

        model_timer = timer()
        parameter_tracker = Track_Weight_Change_onEpoch()

        print(f"STARTING TRAINING: {model_id}, Alpha: {alpha}")
        history = conv_net.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_test, y_test),
            epochs=n_epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[model_timer, parameter_tracker],
        )
        print(f"FINISHED TRAINING: {model_id}")

        # add results to dictionary and store the resulting model weights.
        metrics[model_id] = history.history

        # clear GPU of prior model to decrease training times.
        tf.keras.backend.clear_session()

        # Save results to the data file.
        if save:
            pkl.dump(metrics, open(data_save_path, "wb"))
            history.model.save_weights(model_weights_paths + model_id)

    return metrics

def rand_crop(image, padded_height=36, padded_width=36, crop_dim=32):
    """
       Helper function to apply random cropping to the training data set via tf.keras.preprocessing.image.ImageDataGenerator.
    """
    height_pad = padded_height-crop_dim
    width_pad = padded_width-crop_dim
    
    offset_height = tf.random.uniform([], 0, padded_height, dtype=tf.int32).numpy()
    offset_width = tf.random.uniform([], 0, width_pad, dtype=tf.int32).numpy()
    
    image = tf.image.pad_to_bounding_box(image, height_pad // 2, width_pad // 2, padded_height, padded_width)
    return tf.image.random_crop(image, size=[crop_dim, crop_dim, image.shape[2]])


def train_resnet18(
    data_set,
    resnet_widths,
    label_noise_as_int=10,
    scaled_loss_alpha=None,
    n_epochs=None,
    n_batch_steps=500_000,
    optimizer=None,
    save=True,
    data_save_path_prefix="",
    data_save_path_suffix="",
    load_saved_metrics=False
):
    """
    Train and save the results of ResNets nets of a given range of model widths.

    Parameters
    ----------
    data_set: str
        Which data set to train on. See the load data funciton.
    resnet_widths: list[int]
        List of model widths to train.
    label_noise_as_int: int
        Percentage of label noise to add to the training data.
    scaled_loss_alpha: float
        The alplha value used to scale the cross entropy loss used during training.
    n_epochs: int
        number of epochs to train, if not specified, will calculate with n_batch_steps
    n_batch_steps: int
        number of gradient descent steps to take, over-ridden if n_epochs is specified
    optimizer: tf.keras.optimizer
        Optimizer to use while training resnets. Default is Adam with a learning rate of 1e-4.
    save: bool
        whether to save the data and trained model weights.
    data_save_path_prefix: str
        prefix to add to the save pkl file path.
    data_save_path_suffix: str
        suffix to add to the save pkl file name.
    load_saved_metrics: bool
        if True, will attempt to load the metrics from a previous training session in the save_path,
        to continue training from there. If True, will load the saved .pkl file instead of starting
        over and overwriting it. 
    """

    alpha = scaled_loss_alpha if scaled_loss_alpha is not None else 1
    label_noise = label_noise_as_int / 100

    # load the relevent dataset
    (x_train, y_train), (x_test, y_test), image_shape = load_data(data_set, label_noise)

    batch_size = 128
    n_classes = tf.math.reduce_max(y_train).numpy() + 1

    # total number desirec SGD steps / number batches per epoch = n_epochs
    if not n_epochs:
        n_epochs = n_batch_steps // (x_train.shape[0] // batch_size)

    # store results for later graphing and analysis.
    model_histories = {}
    metrics = {}

    # Paths to save model weights and experimental results.
    model_weights_paths = f"trained_model_weights_{data_set}/resnet18_{label_noise_as_int}pct_noise_alpha_{alpha}/"
    data_save_path = (
        f"experimental_results_{data_set}/resnet18_{label_noise_as_int}pct_noise_alpha_{alpha}".replace(
            ".", "_"
        )
        + ".pkl"
    )

    # add possible path identifiers.
    if data_save_path_prefix:
        model_weights_paths = data_save_path_prefix + '/' + model_weights_paths
        data_save_path = data_save_path_prefix + '/' + data_save_path

    if data_save_path_suffix:
        assert data_save_path[-4:] == ".pkl"
        data_save_path = data_save_path[:-4] + data_save_path_suffix + ".pkl"

    if load_saved_metrics:
        try:
            with open(data_save_path, 'rb') as f:
                metrics = pkl.load(f)
        except Exception as e:
            print('Could not find saved metrics.pkl file, exiting')
            raise e

        loaded_widths = [int(i.split('_')[-1]) for i in metrics.keys()]
        assert convnet_widths[:len(loaded_widths)] == loaded_widths
        print('loaded results for width %s from existing file at %s' %(', '.join([str(i) for i in loaded_widths]), data_save_path))

        assert data_save_path[-4:] == ".pkl"
        data_backup_path = data_save_path[:-4] + 'backup_w%d_' %loaded_widths[-1] + time.strftime("%D_%H%M%S").replace('/', '') + ".pkl"
        print('saving existing result.pkl to backup at %s' %data_backup_path)
        pkl.dump(metrics, open(data_backup_path, "wb"))

    for width in resnet_widths:
        if scaled_loss_alpha is None:
            scaled_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        elif scaled_loss_alpha=='1_m' or scaled_loss_alpha=='1_sqrtm':
            scaled_loss = get_scaled_sparse_categorical_loss(scaled_loss_alpha, width)
        else:
            scaled_loss = get_scaled_sparse_categorical_loss(scaled_loss_alpha)

        if load_saved_metrics and width in loaded_widths:
            print('width %d results already loaded from .pkl file, training skipped' %width)
            continue

        # Resnet18 with Kaiming Uniform Initialization.
        resnet, model_id = make_resnet18_UniformHe(
            image_shape, k=width, num_classes=n_classes
        )

        # compile and pass input to initialize parameters.
        resnet.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4, epsilon=1e-08)
            if optimizer is None
            else optimizer,
            loss=scaled_loss,
            metrics=["accuracy"],
        )
        resnet(tf.keras.Input(shape=list(image_shape), batch_size=batch_size))

        model_timer = timer()
        parameter_tracker = Track_Weight_Change_onEpoch()

        print(f"STARTING TRAINING: {model_id}, Alpha: {alpha}")
        history = resnet.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_test, y_test),
            epochs=n_epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[model_timer, parameter_tracker],
        )
        print(f"FINISHED TRAINING: {model_id}")

        # add results to dictionary and store the resulting model weights.
        metrics[model_id] = history.history

        # clear GPU of prior model to decrease training times.
        tf.keras.backend.clear_session()

        # Save results to the data file
        if save:
            pkl.dump(metrics, open(data_save_path, "wb"))
            history.model.save_weights(model_weights_paths + model_id)

    return metrics


def get_scaled_sparse_categorical_loss(alpha=1, width=None):
    """
    Create a Custom Loss function which is equivalent to rescalling the initialization variance.

    See numerical experiments in the paper:
    On Lazy Training in Differentiable Programming, Chizat et. al. 2020
    (https://arxiv.org/pdf/1812.07956.pdf)
    """
    if alpha == '1_m' or alpha == '1_sqrtm':
        if not width:
            raise Exception('width must be specified when using alpha proportional to width')

        scaled_alpha = 1/width if alpha == '1_m' else 1/(width**0.5)
        def scaled_sparse_categorical_loss(y_actual, y_pred):
            sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            scaled_sce = sce(y_actual, scaled_alpha * y_pred) / scaled_alpha ** 2
            return scaled_sce     

        return scaled_sparse_categorical_loss

    else:         
        const_alpha = alpha
        def scaled_sparse_categorical_loss(y_actual, y_pred):
            sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            scaled_sce = sce(y_actual, const_alpha * y_pred) / const_alpha ** 2
            return scaled_sce

        return scaled_sparse_categorical_loss


def load_data(data_set, label_noise):
    """
    Helper Function to Load data in the form of a tensorflow data set, apply label noise, and return the
    train data and test data.

    Parameters
    ----------
    data_set: str
        name of data set to load from tf.keras.datasets
    label_noise: float
        percentage of training data to add noise to
    """

    datasets = ["cifar10", "cifar100", "mnist"]

    # load Cifar 10, Cifar 100, or mnis data set
    if data_set == "cifar10":
        get_data = tf.keras.datasets.cifar10
    elif data_set == "cifar100":
        get_data = tf.keras.datasets.cifar100
    elif data_set == "mnist":
        get_data = tf.keras.datasets.mnist
    else:
        raise Exception(
            f"Please enter a data set from the following options: {datasets}"
        )

    # load the data.
    (x_train, y_train), (x_test, y_test) = get_data.load_data()
    image_shape = x_train[0].shape
    
    # apply label noise to the data set
    if 0 < label_noise:
        random_idx = np.random.choice(
            x_train.shape[0], int(label_noise * x_train.shape[0])
        )
        rand_labels = np.random.randint(
            low=y_train.min(), high=y_train.max(), size=len(random_idx)
        )
        y_train[random_idx] = np.expand_dims(rand_labels, axis=1)

    # cast values to tf.float32 and normalize images to range [0-1]
    x_train, x_test = (
        tf.cast(x_train, tf.float32) / 255,
        tf.cast(x_test, tf.float32) / 255,
    )
    y_train, y_test = tf.cast(y_train, tf.float32), tf.cast(y_test, tf.float32)

    return (x_train, y_train), (x_test, y_test), list(image_shape)


class Track_Weight_Change_onEpoch(tf.keras.callbacks.Callback):
    """
    Tensorflow Call back to track the L2 norm in the difference between the intial model weights
    and the model weights at the end of each epoch.

    We only track the change in the kernels of convolutional and dense layers.
    """

    def __init__(self):
        super().__init__()
        self.initial_weights = None

    def on_train_begin(self, logs=None):
        # store the models initialized weights. Need to cast to numpy, otherwise, the values change during training!
        self.initial_weights = [
            layer.weights[0].numpy()
            for layer in self.model.layers
            if (
                isinstance(layer, tf.keras.layers.Conv2D)
                or isinstance(layer, tf.keras.layers.Dense)
            )
        ]
        self.model.history.history["weight_change_l2"] = []
        self.model.history.history["weight_change_inf"] = []

    def on_epoch_end(self, epoch, logs=None):
        # Calculate the Norm of the difference between the current layer parameters and the inialized parameters.
        curr_weights = [
            layer.weights[0].numpy()
            for layer in self.model.layers
            if (
                isinstance(layer, tf.keras.layers.Conv2D)
                or isinstance(layer, tf.keras.layers.Dense)
            )
        ]

        # change l2 norm
        weight_change_l2 = [
            tf.norm(init_layer - curr_layer, ord=2).numpy()
            for init_layer, curr_layer in zip(self.initial_weights, curr_weights)
        ]

        # change infinity norm
        weight_change_inf = [
            tf.norm(init_layer - curr_layer, ord=np.inf).numpy()
            for init_layer, curr_layer in zip(self.initial_weights, curr_weights)
        ]

        self.model.history.history["weight_change_l2"].append(weight_change_l2)
        self.model.history.history["weight_change_inf"].append(weight_change_inf)


class timer(tf.keras.callbacks.Callback):
    """
    Simle call back class to track total training time.
    """

    def __init__(self, n_epochs=25):
        super().__init__()

        self.start_time = time.perf_counter()
        self.n_epochs = n_epochs

    def on_epoch_end(self, epoch, logs=None):
        """ Help keep track of total training time needed for various models. """
        if epoch % self.n_epochs == 0:
            end_time = time.perf_counter()
            run_time = end_time - self.start_time
            hrs, mnts, secs = (
                int(run_time // 60 // 60),
                int(run_time // 60 % 60),
                int(run_time % 60),
            )

            template = "Epoch: {:04}, Total Run Time: {:02}:{:02}:{:02}"
            template += " - Loss: {:.4e}, Accuracy: {:.3f}, Test Loss: {:.4e}, Test Accuracy: {:.3f}"

            train_loss, train_accuracy = logs["loss"], logs["accuracy"]
            test_loss, test_accuracy = logs["val_loss"], logs["val_accuracy"]
            print(
                template.format(
                    epoch,
                    hrs,
                    mnts,
                    secs,
                    train_loss,
                    train_accuracy,
                    test_loss,
                    test_accuracy,
                )
            )


class inverse_squareroot_lr:
    """
    This is the learning rate used with SGD in the paper (Inverse square root decay).
    Learning Rate starts at 0.1 and then drops every 512 batches.
    """

    def __init__(self, n_steps=512, init_lr=0.1):
        self.n = n_steps
        self.gradient_steps = 0
        self.init_lr = init_lr

    def __call__(self):
        lr = self.init_lr / tf.math.sqrt(
            1.0 + tf.math.floor(self.gradient_steps / self.n)
        )
        self.gradient_steps += 1
        return lr


class Model_Trainer:
    # Training Wrapper For Tensorflow Models. Allows a predifined model to be easily trained
    # while also tracking parameter and gradient information.

    # Please ensure that model_id is unique. It provides the path for all model statistics.

    """
    NO longer in use, model.fit provides significant training speed up. the features utilized below
    will be replaced with tensorflow callbacks.
    """

    def __init__(
        self, model, model_id, lr=1e-4, optimizer=None, data_augmentation=None
    ):
        """
        Parameters
        ----------

        model: tensorflow.keras.Model
        model_id : string
            An identifying string used in saving model metrics.
        lr : float, tensorflow.keras.optimizers.schedules
            If using the default optimizer, this is the lr used in the Adam optimizer.
            This value is ignored if an optimizer is passed to the trainer.
        optimizer : tensorflow.keras.optimizers
            A pre-defined optimizer used in training the neural network
        data_augmentation : tensorflow.keras.Sequential
            A tensorflow model used to perform data augmentation during training.
            See here: https://www.tensorflow.org/tutorials/images/data_augmentation#use_keras_preprocessing_layers
        """

        self.lr = lr

        self.model = model
        self.init_loss()

        # Can optionally pass a seperate optimizer.
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.init_optimizer()

        if data_augmentation is not None:
            self.is_data_augmentation = True
            self.data_augmentation = data_augmentation
        else:
            self.is_data_augmentation = False
            self.data_augmentation = None

        # Used to save the parameters of the model at a given point of time.
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self.checkpoint_path = (
            self.model.__class__.__name__ + "/" + model_id + "/training_checkpoints"
        )

        self.summary_path = (
            self.model.__class__.__name__ + "/" + model_id + "/summaries/"
        )
        self.summary_writer = tf.summary.create_file_writer(self.summary_path)

        self.gradients = None

    # initialize loss function and metrics to track over training
    def init_loss(self):
        self.loss_function = SparseCategoricalCrossentropy()

        self.train_loss = Mean(name="train_loss")
        self.train_accuracy = SparseCategoricalAccuracy(name="train_accuracy")

        self.test_loss = Mean(name="test_loss")
        self.test_accuracy = SparseCategoricalAccuracy(name="test_accuracy")

    # Initialize Model optimizer
    def init_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, epsilon=1e-8)

    # Take a single Training step on the given batch of training data.
    def train_step(self, images, labels, track_gradient=False):

        with tf.GradientTape() as gtape:
            predictions = self.model(images, training=True)
            loss = self.loss_function(labels, predictions)

        gradients = gtape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Track model Performance
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

        return self.train_loss.result(), self.train_accuracy.result() * 100

    # Evaluate Model on Test Data
    def test_step(self, data_set):
        predictions = self.model.predict(images)
        test_loss = self.loss_function(labels, predictions)

        self.test_loss(test_loss)
        self.test_accuracy(labels, predictions)

        return self.test_loss.result(), self.test_accuracy.result() * 100

    # Reset Metrics
    def reset(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()

        self.test_loss.reset_states()
        self.test_accuracy.reset_states()

    # Save a checkpoint instance of the model for later use
    def model_checkpoint(self):
        # Save a checkpoint to /tmp/training_checkpoints-{save_counter}
        save_path = self.checkpoint.save(self.checkpoint_path)
        return save_path

    def log_metrics(
        self,
    ):
        # Log metrics using tensorflow summary writer. Can Then visualize using TensorBoard
        step = self.checkpoint.save_counter

        with self.summary_writer.as_default():
            tf.summary.scalar("Train Loss", self.train_loss.result(), step=step)
            tf.summary.scalar("Train Accuracy", self.train_accuracy.result(), step=step)
            tf.summary.scalar("Test Loss", self.test_loss.result(), step=step)
            tf.summary.scalar("Test Accuracy", self.test_accuracy.result(), step=step)
