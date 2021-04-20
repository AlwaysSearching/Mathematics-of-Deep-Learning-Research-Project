import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import Dense, Flatten 
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import pickle as pkl
import time

DATA = 'CIFAR10'
NORMALIZE = True

if DATA == 'MNIST':
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
if DATA == 'CIFAR10':
  cifar10 = tf.keras.datasets.cifar10
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
if NORMALIZE:
  x_train, x_test = x_train / 255.0, x_test / 255.0

class CustomCallback_epoch(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = time.perf_counter()

    def on_train_begin(self, logs=None):
        global weight_history
        # print('SAVING INITIAL WEIGHT VALUES')
        weight_history.append(self.model.get_weights())

    def on_epoch_end(self, epoch, logs=None):
        global weight_history
        # list of weight tensors
        curr_weight = self.model.get_weights()
        if weight_history:
          weight_change = [curr_weight[i] - weight_history[0][i] for i in range(len(curr_weight))]
          norm_delta = [tf.norm(t, ord=2).numpy() for t in weight_change]
          # print('L2 NORM OF WEIGHT CHANGE RELATIVE TO INITIAL VALUES: ', norm_delta)
          weight_history.append(norm_delta)

        end_time = time.perf_counter()
        run_time = end_time - self.start_time
        hrs, mnts, secs = int(run_time // 60 // 60), int(run_time // 60 % 60), int(run_time % 60)

        template = 'Epoch: {:04}, Total Run Time: {:02}:{:02}:{:02}'
        template += ' - Loss: {:.4e}, Accuracy: {:.3f}, Test Loss: {:.4e}, Test Accuracy: {:.3f}'
        template += ' - L2 Norm of Weight Movement From Initialization: %s' %str(norm_delta)

        train_loss, train_accuracy = logs['loss'], logs['accuracy']
        test_loss, test_accuracy = logs['val_loss'], logs['val_accuracy']
        print(template.format(epoch, hrs, mnts, secs, train_loss, train_accuracy, test_loss, test_accuracy), end='\r')


# some parameters in the active/lazy paper:
# optimizer: sgd with momentum=0.9
# initialization: kernel - xavier(glorot) normal, bias - zeros (default for Dense in tf)
# loss: both categroical crossentropy(sparse) and mse are provided as options, using cce here
# scaling: implemented in the paper repo's train.py as loss = torch.nn.CrossEntropyLoss(alpha*outputs, targets)/alpha**2
#          implemented here with scaled_custom_loss() loss function that does the same thing        

def train(alpha, epoch, opt, lr, scaling=True, width=64):
    global weight_history
    weight_history = []

    def scaled_custom_loss(y_actual, y_pred):
        sce = tf.keras.losses.SparseCategoricalCrossentropy()
        scaled_sce = sce(y_actual, alpha*y_pred)/alpha**2
        return scaled_sce

    if DATA == 'CIFAR10':
      flatten = tf.keras.layers.Flatten(input_shape=(32, 32, 3))

    if DATA == 'MNIST':
      flatten = tf.keras.layers.Flatten(input_shape=(28, 28))

    model = tf.keras.models.Sequential([
      flatten,
      tf.keras.layers.Dense(width, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    # model.summary()
  
    if opt=='adam':
      # default lr for Adam is 0.001
      opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt=='sgd':
      # default lr for sgd is 0.01
      opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
      raise Exception('optimizer must be adam/sgd')

    model.compile(optimizer=opt,
                  loss=scaled_custom_loss if scaling else 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=epoch, validation_data=(x_test, y_test), callbacks=[CustomCallback_epoch()], verbose=0)
    
    print()
    print('max training accuracy', max(history.history['accuracy']))
    print('min training loss', min(history.history['loss']))
    print('max validation accuracy', max(history.history['val_accuracy']))
    print('min validation loss', min(history.history['val_loss']))

    # plot(history)
    # print(tf.math.confusion_matrix(y_test, tf.argmax(model.predict(x_test), axis=1)))
    print()
    print('l2-normed weight changes from initial values after last epoch:')
    print(weight_history[-1])

    return (weight_history[1:], history.history)


lrs = [1.0, 0.1, 0.01, 0.001]
alphas = [20000.0, 1000.0, 100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.07, 0.04, 0.01, 0.007, 0.003, 0.001]
normed_weight_changes_w64 = {}
optimizer = 'sgd'
num_epochs = 10
for learning_rate in lrs:
  for alpha_val in alphas:
    print('='*80)
    print('w64 opt = %s, lr = %f, alpha = %f' %(optimizer, learning_rate, alpha_val))
    print('='*80)
    normed_weight_changes_w64[(optimizer, learning_rate, alpha_val)] = train(alpha=alpha_val, epoch=num_epochs, opt=optimizer, lr=learning_rate, scaling=True)
  pkl.dump(normed_weight_changes_w64, open('normed_weight_changes_w64.pkl', "wb"))   


lrs = [1.0, 0.1, 0.01, 0.001]
alphas = [20000.0, 1000.0, 100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.07, 0.04, 0.01, 0.007, 0.003, 0.001]
normed_weight_changes_w256 = {}
optimizer = 'sgd'
num_epochs = 10
for learning_rate in lrs:
  for alpha_val in alphas:
    print('='*80)
    print('w256 opt = %s, lr = %f, alpha = %f' %(optimizer, learning_rate, alpha_val))
    print('='*80)
    normed_weight_changes_w256[(optimizer, learning_rate, alpha_val)] = train(alpha=alpha_val, epoch=num_epochs, opt=optimizer, lr=learning_rate, scaling=True, width=256)
  pkl.dump(normed_weight_changes_w256, open('normed_weight_changes_w256.pkl', "wb"))   


# experiment with default network size (width=64) but using adam optimizer
lrs = [1.0, 0.1, 0.01, 0.001]
alphas = [20000.0, 1000.0, 100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.07, 0.04, 0.01, 0.007, 0.003, 0.001]
normed_weight_changes_adam = {}
optimizer = 'adam'
num_epochs = 10
for learning_rate in lrs:
  for alpha_val in alphas:
    print('='*80)
    print('adam opt = %s, lr = %f, alpha = %f' %(optimizer, learning_rate, alpha_val))
    print('='*80)
    normed_weight_changes_adam[(optimizer, learning_rate, alpha_val)] = train(alpha=alpha_val, epoch=num_epochs, opt=optimizer, lr=learning_rate, scaling=True)
  pkl.dump(normed_weight_changes_adam, open('normed_weight_changes_adam.pkl', "wb"))   


# experiment with default network size (width=64) but more epochs (100)
lrs = [1.0, 0.1, 0.01, 0.001]
alphas = [20000.0, 1000.0, 100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.07, 0.04, 0.01, 0.007, 0.003, 0.001]
normed_weight_changes_e100 = {}
optimizer = 'sgd'
num_epochs = 50
for learning_rate in lrs:
  for alpha_val in alphas:
    print('='*80)
    print('e50 opt = %s, lr = %f, alpha = %f' %(optimizer, learning_rate, alpha_val))
    print('='*80)
    normed_weight_changes_e50[(optimizer, learning_rate, alpha_val)] = train(alpha=alpha_val, epoch=num_epochs, opt=optimizer, lr=learning_rate, scaling=True)
  pkl.dump(normed_weight_changes_e50, open('normed_weight_changes_e100.pkl', "wb"))   
