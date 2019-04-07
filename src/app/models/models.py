from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras


def __get_ml_model_details():
    return [
        {
            'Name': 'NeuralNet with 2 Layers (128, 10 nodes each) and Adam optimizer',
            'Type': 'NeuralNet',
            'Layers': [
                {
                    'HiddenLayer': 1,
                    'NumberOfNodes': 128,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'HiddenLayer': 2,
                    'NumberOfNodes': 10,
                    'ActivationFunction': tf.nn.relu
                }
            ],
            'Optimizer': 'adam',
            'Loss': 'sparse_categorical_crossentropy',
            'Metrics': ['accuracy']
        },
        {
            'Name': 'NeuralNet with 3 Layers (128 relu, 10 relu, 10 sigmoid nodes each) and NADAM optimizer',
            'Type': 'NeuralNet',
            'Layers': [
                {
                    'HiddenLayer': 1,
                    'NumberOfNodes': 128,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'HiddenLayer': 2,
                    'NumberOfNodes': 10,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'HiddenLayer': 3,
                    'NumberOfNodes': 10,
                    'ActivationFunction': tf.nn.sigmoid
                }
            ],
            'Optimizer': 'nadam',
            'Loss': 'sparse_categorical_crossentropy',
            'Metrics': ['accuracy']
        }
    ]


def get_ml_models():
    built_models = __get_ml_model_details()
    for model_detail in built_models:
        # Build each model using the details and store the model in the details
        model_detail['Model'] = __build_model(model_detail)

    print(built_models)
    return built_models


def __build_model(meta_model={}):
    model = keras.Sequential()
    for layer in meta_model['Layers']:
        model.add(keras.layers.Dense(layer['NumberOfNodes'], activation=layer['ActivationFunction']))

    model.compile(optimizer=meta_model['Optimizer'],
                  loss=meta_model['Loss'],
                  metrics=meta_model['Metrics'])

    return model
