import tensorflow as tf

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
from tensorflow_addons.metrics import MultiLabelConfusionMatrix

class Model_Trainer:
    # Training Wrapper For Tensorflow Models. Allows a predifined model to be easily trained
    # while also tracking parameter and gradient information.
    
    def __init__(self, model, lr=5e-4, optimizer=None):
                
        self.lr = lr
        self.n_classes = Model.n_classes       
        
        self.model = Model
        self.init_loss()

        # Can optionally pass a seperate optimizer.
        if optimizer is None:
            self.optimizer = optimizer
        
        # Used to save the parameters of the model at a given point of time.
        self.checkpoint = tf.train.Checkpoint(self.model)
        self.checkpoint_path = self.model.__class__.__name__ + "/training_checkpoints"


        self.summary_path = self.model.__class__.__name__ + '/summaries/'
        self.train_summary_writer = tf.summary.create_file_writer(self.summary_path + 'train')
        self.test_summary_writer = tf.summary.create_file_writer(self.summary_path + 'test')
        
        self.gradients
        
    
    #initialize loss function and metrics to track over training
    def init_loss(self):
        self.loss_function = SparseCategoricalCrossentropy(from_logits=True)

        self.train_loss = Mean(name='train_loss')
        self.train_accuracy = SparseCategoricalAccuracy(name='train_accuracy')
        self.train_confusion = MultiLabelConfusionMatrix(num_classes=self.n_classes, name='train_confusion_matrix')

        self.test_loss = Mean(name='test_loss')
        self.test_accuracy = SparseCategoricalAccuracy(name='test_accuracy')
        self.test_confusion = MultiLabelConfusionMatrix(num_classes=self.n_classes, name='test_confusion_matrix')
        

    # Initialize Model optimizer
    def init_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
    
    # Take a single Training step on the given batch of training data.
    @tf.function
    def train_step(self, images, labels, track_gradient=False):
        with tf.GradientTape() as gtape:
            predictions = self.model(images, training=True)
            loss = self.loss_function(labels, predictions)
            
        gradients = gtape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Track Gradient Information
        
        # Track model Performance
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
        self.train_confusion(labels, predictions)
        
        return self.train_loss.result(), self.train_accuracy.result()*100, self.train_confusion.result()
    
    # Evaluate Model on Test Data
    @tf.function
    def test_step(self, images, labels):
        predictions = self.model.predict(images)
        test_loss = self.loss_function(labels, predictions)
        
        self.test_loss(test_loss)
        self.test_accuracy(labels, predictions) 
        self.test_confusion(labels, predictions)
        
        return self.test_loss.result(), self.test_accuracy.result()*100, self.test_confusion.result()
        
    # Reset Metrics 
    @tf.function
    def reset(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.train_confusion.reset()
        
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        self.test_confusion.reset()
        
    # Save a checkpoint instance of the model for later use
    def model_checkpoint(self):
        # Save a checkpoint to /tmp/training_checkpoints-{save_counter}
        save_path = checkpoint.save(self.checkpoint_path)
        return save_path

    def log_metrics(self, step):
        # Log metrics using tensorflow summary writer. Can Then visualize using TensorBoard
        with self.train_summary_writer.as_default():
            tf.summary.scalar('Train Loss', self.train_loss.result(), step=step)
            tf.summary.scalar('Train Accuracy', self.train_accuracy.result(), step=step)
            tf.summary.scalar('Train Confusion Matrix', self.train_confusion.result(), step=step)

        with self.test_summary_writer.as_default():
            tf.summary.scalar('Test Loss', self.test_loss.result(), step=step)
            tf.summary.scalar('Test Accuracy', self.test_accuracy.result(), step=step)
            tf.summary.scalar('Test Confusion Matrix', self.test_confusion.result(), step=step)
