from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

import data
import numpy

# Get training data/labels, test data/labels from data as numpy arrays
((train_data, train_labels), (test_data, test_labels)) = data.get_data()

# Build the model
model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.relu)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
print("Training the model")
model.fit(train_data, train_labels, epochs=10)
print("Trained")

# Test the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)

predictions = model.predict(test_data)

print(numpy.argmax(predictions[0]))
print(test_labels[0])
