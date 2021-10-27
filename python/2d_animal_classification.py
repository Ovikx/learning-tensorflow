import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

data = pd.read_csv('datasets/2D_classification_animals.csv')

species_names = {
    0 : 'Kudu',
    1 : 'Gorilla'
}

mass_data = data['Mass'].to_list()
height_data = data['Height'].to_list()
features = np.array([(mass_data[i], height_data[i]) for i in range(len(mass_data))]).reshape(-1,2)
labels = np.array([0 if v == 'Kudu' else 1 for v in data['Species'].to_list()]).reshape(-1,1)

train_threshold = int(len(features)*0.7)

train_features = features[:train_threshold]
train_labels = labels[:train_threshold]

test_features = features[train_threshold:]
test_labels = labels[train_threshold:]

train_ds = tf.data.Dataset.from_tensors((train_features, train_labels))
test_ds = tf.data.Dataset.from_tensors((test_features, test_labels))

model = keras.models.Sequential([
    keras.layers.Dense(8, input_dim=2),
    keras.layers.Dense(1, activation='sigmoid') #last layer is non-linear
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1/2), loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
model.fit(train_ds, epochs=200)

preds = [v[0] for v in (model.predict(test_features)).tolist()]
test_results = [species_names[round(v)] == species_names[test_labels[i][0]] for i, v in enumerate(preds)]
print(test_results.count(True)/len(test_results))

print(model.predict([[270, 1.4]]))