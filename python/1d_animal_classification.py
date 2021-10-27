import tensorflow as tf
from tensorflow import keras
import pandas as pd

data = pd.read_csv("datasets/1D_classification_animals.csv")

species_names = {
    0 : 'Kudu',
    1 : 'Gorilla'
}

features = data['Mass'].to_list()
labels = [0 if v == 'Kudu' else 1 for v in data['Species'].to_list()]

train_threshold = int(len(features)*0.7)

train_features = features[:train_threshold]
train_labels = labels[:train_threshold]

test_features = features[train_threshold:]
test_labels = labels[train_threshold:]

model = keras.models.Sequential([
    keras.layers.Dense(1, input_dim=1),
    keras.layers.Dense(1, activation='sigmoid') #last layer is non-linear
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_features, train_labels, epochs=200)

preds = [v[0] for v in (model.predict(test_features)).tolist()]

test_results = [species_names[round(v)] == species_names[test_labels[i]] for i, v in enumerate(preds)]
print(test_results.count(True)/len(test_results))

print(model.predict([170, 250]))