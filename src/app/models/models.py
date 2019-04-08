from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# An config for different ML Models
ml_models_metadata = [
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
                },
                {
                    'NumberOfNodes': 1,
                    'ActivationFunction': tf.nn.sigmoid
                }
            ],
            'Optimizer': 'adam',
            'Loss': 'binary_crossentropy',
            'Metrics': ['accuracy']
        },
        {
            'Name': 'NeuralNet with 2 Layers (128, 10 nodes each) and NADAM optimizer',
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
                    'NumberOfNodes': 1,
                    'ActivationFunction': tf.nn.sigmoid
                }
            ],
            'Optimizer': 'nadam',
            'Loss': 'binary_crossentropy',
            'Metrics': ['accuracy']
        },
        {
            'Name': 'NeuralNet with 3 Layers (128 relu, 10 relu, 10 relu nodes each) and Adam optimizer',
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
                    'HiddenLayer': 2,
                    'NumberOfNodes': 10,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'NumberOfNodes': 1,
                    'ActivationFunction': tf.nn.sigmoid
                }
            ],
            'Optimizer': 'adam',
            'Loss': 'binary_crossentropy',
            'Metrics': ['accuracy']
        },
        {
            'Name': 'NeuralNet with 3 Layers (128 relu, 10 relu, 10 relu nodes each) and NADAM optimizer',
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
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'NumberOfNodes': 1,
                    'ActivationFunction': tf.nn.sigmoid
                }
            ],
            'Optimizer': 'nadam',
            'Loss': 'binary_crossentropy',
            'Metrics': ['accuracy']
        }
    ]

"""
Gets all the ML Model meta data needed to construct the models, constructs the appropriate ML model, and adds it as part of the 
metadata array. Returns this list of ml model metadata that now has the compiled ML models as well
"""
def get_ml_models():
    built_models = __get_ml_model_details()
    for model_detail in built_models:
        # Build each model using the details and store the model in the details
        model_detail['Model'] = __build_model(model_detail)

    return built_models


def __get_ml_model_details():
    return ml_models_metadata.copy()


def __build_model(meta_model={}):
    model = None
    if meta_model['Type'] == 'NeuralNet':
        model = keras.Sequential()
        for layer in meta_model['Layers']:
            model.add(keras.layers.Dense(layer['NumberOfNodes'], activation=layer['ActivationFunction']))

        model.compile(optimizer=meta_model['Optimizer'],
                    loss=meta_model['Loss'],
                    metrics=meta_model['Metrics'])

    return model
