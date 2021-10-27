import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import pandas as pd
import numpy as np

data = pd.read_csv('datasets/2D_classification_animals.csv')

species_names = {
    0 : 'Kudu',
    1 : 'Gorilla'
}

# Separate columns
mass_data = data['Mass'].to_list()
height_data = data['Height'].to_list()

# Turn features/labels into reshaped numpy arrays
features = np.array([(mass_data[i], height_data[i]) for i in range(len(mass_data))]).reshape(-1,2)
labels = np.array([0 if v == 'Kudu' else 1 for v in data['Species'].to_list()]).reshape(-1,1)

# Figure out how much of the data should be used for training
train_threshold = int(len(features)*0.7)

# Partition the data
train_features = features[:train_threshold]
train_labels = labels[:train_threshold]

test_features = features[train_threshold:]
test_labels = labels[train_threshold:]

# Prepare data for training/testing
train_ds = tf.data.Dataset.from_tensors((train_features, train_labels))
test_ds = tf.data.Dataset.from_tensors((test_features, test_labels))

# Neural network class
class AnimalPredictor(Model):
    # Define the layers in the constructor
    def __init__(self):
        super(AnimalPredictor, self).__init__()
        self.d1 = Dense(8, input_shape=(2,), name='dense') # Make sure to define the input shape
        self.d2 = Dense(1, activation='sigmoid', name='output')

    # Pass the inputs through the layers
    def call(self, x):
        x = self.d1(x)
        return self.d2(x)

# Create the NN
model = AnimalPredictor()

loss_object = keras.losses.BinaryCrossentropy(from_logits=False) # logits map [0, 1] values to (-inf, inf)
optimizer = keras.optimizers.Adam(learning_rate=0.1)

train_loss = keras.metrics.Mean(name='train_loss')
train_accuracy = keras.metrics.BinaryAccuracy(name='train_accuracy')

test_loss = keras.metrics.Mean(name='test_loss')
test_accuracy = keras.metrics.BinaryAccuracy(name='test_accuracy')

@tf.function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(features, labels):
    predictions = model(features, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

for epoch in range(200):
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

print(model.predict([[270, 1.4]]))