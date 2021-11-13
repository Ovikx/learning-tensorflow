import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model
import numpy as np
from image_preprocessor import ImagePreprocessor
np.set_printoptions(suppress=True)

flavors = {
    0 : 'chocolate',
    1 : 'vanilla'
}
pix = 64

ip = ImagePreprocessor(normalization=255, training_threshold=0.8, color_mode='RGB')
package = ip.preprocess_dirs(['cnn/images/chocolate', 'cnn/images/vanilla'], [0, 1], True)

train_features = package['TRAIN_IMAGES']
train_labels = package['TRAIN_LABELS']
test_features = package['TEST_IMAGES']
test_labels = package['TEST_LABELS']

train_ds = tf.data.Dataset.from_tensors((train_features, train_labels)).shuffle(10000)
test_ds = tf.data.Dataset.from_tensors((test_features, test_labels)).shuffle(10000)

class FlavorPredictor(Model):
    def __init__(self):
        super(FlavorPredictor, self).__init__()
        self.conv1 = Conv2D(16, (3,3), activation='relu', input_shape=(pix,pix,3))
        self.mp1 = MaxPool2D()
        self.conv2 = Conv2D(32, (3,3), activation='relu')
        self.mp2 = MaxPool2D()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(1, activation='sigmoid')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = FlavorPredictor()

loss_function = keras.losses.BinaryCrossentropy(from_logits=False)
optimizer = keras.optimizers.Adam(learning_rate=0.001)

train_loss = keras.metrics.Mean()
train_accuracy = keras.metrics.BinaryAccuracy(name='train_accuracy')

test_loss = keras.metrics.Mean()
test_accuracy = keras.metrics.BinaryAccuracy()

@tf.function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_function(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(features, labels):
    predictions = model(features, training=False)
    t_loss = loss_function(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

for epoch in range(100):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for features, label in train_ds:
        train_step(features, label)

    for features, label in test_ds:
        test_step(features, label)
    
    print(
        f'Epoch {epoch+1} || '
        f'Training Loss: {train_loss.result()}, '
        f'Training Accuracy: {train_accuracy.result()}, '
        f'Testing Loss: {test_loss.result()}, '
        f'Testing Accuracy: {test_accuracy.result()}'
    )
    
print('EXTERNAL SAMPLE TESTING\nOutputs are [0, 1] where 0 is chocolate and 1 is vanilla\n----------------------')
print(f'Vanilla cupcakes: {model.predict(np.array(ip.file_to_array("cnn/images/external_test/v1.jpg")))}')
print(f'Vanilla ice cream: {model.predict(np.array(ip.file_to_array("cnn/images/external_test/v2.jpg")))}')
print(f'Chocolate brittle: {model.predict(np.array(ip.file_to_array("cnn/images/external_test/c1.jpg")))}')
print(f'Chocolate cookies: {model.predict(np.array(ip.file_to_array("cnn/images/external_test/c2.jpg")))}')