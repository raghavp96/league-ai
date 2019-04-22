from __future__ import absolute_import, division, print_function

# To make sure the same randomness is used for feature  in the hidden layers
# From: https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# An config for different ML Models
ml_models_metadata = [
    {
            'Name': 'NeuralNet with 1 Layers (4 relu nodes) and Adam optimizer',
            'Type': 'NeuralNet',
            'Layers': [
                {
                    'HiddenLayer': 1,
                    'NumberOfNodes': 4,
                    'ActivationFunction': tf.nn.relu
                }
            ],
            'Optimizer': 'adam',
            'Loss': 'binary_crossentropy',
            'Metrics': ['accuracy'],
            'Epochs': 10
        },
        {
            'Name': 'NeuralNet with 2 Layers (4, 2 nodes each) and Adam optimizer',
            'Type': 'NeuralNet',
            'Layers': [
                {
                    'HiddenLayer': 1,
                    'NumberOfNodes': 4,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'HiddenLayer': 2,
                    'NumberOfNodes': 2,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'NumberOfNodes': 1,
                    'ActivationFunction': tf.nn.sigmoid
                }
            ],
            'Optimizer': 'adam',
            'Loss': 'binary_crossentropy',
            'Metrics': ['accuracy'],
            'Epochs': 10
        },
        {
            'Name': 'NeuralNet with 2 Layers (4, 2 nodes each) and NADAM optimizer',
            'Type': 'NeuralNet',
            'Layers': [
                {
                    'HiddenLayer': 1,
                    'NumberOfNodes': 4,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'HiddenLayer': 2,
                    'NumberOfNodes': 2,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'NumberOfNodes': 1,
                    'ActivationFunction': tf.nn.sigmoid
                }
            ],
            'Optimizer': 'nadam',
            'Loss': 'binary_crossentropy',
            'Metrics': ['accuracy'],
            'Epochs': 10
        },
        {
            'Name': 'NeuralNet with 3 Layers (4 relu, 2 relu, 2 relu nodes each) and Adam optimizer',
            'Type': 'NeuralNet',
            'Layers': [
                {
                    'HiddenLayer': 1,
                    'NumberOfNodes': 4,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'HiddenLayer': 2,
                    'NumberOfNodes': 2,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'HiddenLayer': 2,
                    'NumberOfNodes': 2,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'NumberOfNodes': 1,
                    'ActivationFunction': tf.nn.sigmoid
                }
            ],
            'Optimizer': 'adam',
            'Loss': 'binary_crossentropy',
            'Metrics': ['accuracy'],
            'Epochs': 10
        },
        {
            'Name': 'NeuralNet with 3 Layers (4 relu, 2 relu, 2 relu nodes each) and NADAM optimizer',
            'Type': 'NeuralNet',
            'Layers': [
                {
                    'HiddenLayer': 1,
                    'NumberOfNodes': 4,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'HiddenLayer': 2,
                    'NumberOfNodes': 2,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'HiddenLayer': 3,
                    'NumberOfNodes': 2,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'NumberOfNodes': 1,
                    'ActivationFunction': tf.nn.sigmoid
                }
            ],
            'Optimizer': 'nadam',
            'Loss': 'binary_crossentropy',
            'Metrics': ['accuracy'],
            'Epochs': 10
        },
        # {
        #     'Name': 'NeuralNet with 10 Layers (4 or 2 relu per layer) and NADAM optimizer',
        #     'Type': 'NeuralNet',
        #     'Layers': [
        #         {
        #             'HiddenLayer': 1,
        #             'NumberOfNodes': 4,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 2,
        #             'NumberOfNodes': 2,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 3,
        #             'NumberOfNodes': 4,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 4,
        #             'NumberOfNodes': 2,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 5,
        #             'NumberOfNodes': 4,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 6,
        #             'NumberOfNodes': 2,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 7,
        #             'NumberOfNodes': 4,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 8,
        #             'NumberOfNodes': 2,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 9,
        #             'NumberOfNodes': 4,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 10,
        #             'NumberOfNodes': 2,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'NumberOfNodes': 1,
        #             'ActivationFunction': tf.nn.sigmoid
        #         }
        #     ],
        #     'Optimizer': 'nadam',
        #     'Loss': 'binary_crossentropy',
        #     'Metrics': ['accuracy'],
        #     'Epochs': 10
        # },
        # {
        #     'Name': 'NeuralNet with 10 Layers (4 relu per layer) and NADAM optimizer',
        #     'Type': 'NeuralNet',
        #     'Layers': [
        #         {
        #             'HiddenLayer': 1,
        #             'NumberOfNodes': 4,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 2,
        #             'NumberOfNodes': 4,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 3,
        #             'NumberOfNodes': 4,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 4,
        #             'NumberOfNodes': 4,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 5,
        #             'NumberOfNodes': 4,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 6,
        #             'NumberOfNodes': 4,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 7,
        #             'NumberOfNodes': 4,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 8,
        #             'NumberOfNodes': 4,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 9,
        #             'NumberOfNodes': 4,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 10,
        #             'NumberOfNodes': 4,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'NumberOfNodes': 1,
        #             'ActivationFunction': tf.nn.sigmoid
        #         }
        #     ],
        #     'Optimizer': 'nadam',
        #     'Loss': 'binary_crossentropy',
        #     'Metrics': ['accuracy'],
        #     'Epochs': 10
        # },
        # Repeat again with dropout 0
        {
            'Name': 'NeuralNet with 2 Layers (4, 2 nodes each) and Adam optimizer and increasing dropout layers',
            'Type': 'NeuralNet',
            'Layers': [
                {
                    'HiddenLayer': 1,
                    'NumberOfNodes': 4,
                    'Dropout': 0.1,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'HiddenLayer': 2,
                    'NumberOfNodes': 2,
                    'Dropout': 0.2,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'NumberOfNodes': 1,
                    'ActivationFunction': tf.nn.sigmoid
                }
            ],
            'Optimizer': 'adam',
            'Loss': 'binary_crossentropy',
            'Metrics': ['accuracy'],
            'Epochs': 10
        },
        {
            'Name': 'NeuralNet with 2 Layers (4, 2 nodes each) and NADAM optimizer and increasing dropout layers',
            'Type': 'NeuralNet',
            'Layers': [
                {
                    'HiddenLayer': 1,
                    'NumberOfNodes': 4,
                    'Dropout': 0.1,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'HiddenLayer': 2,
                    'NumberOfNodes': 2,
                    'Dropout': 0.2,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'NumberOfNodes': 1,
                    'ActivationFunction': tf.nn.sigmoid
                }
            ],
            'Optimizer': 'nadam',
            'Loss': 'binary_crossentropy',
            'Metrics': ['accuracy'],
            'Epochs': 10
        },
        {
            'Name': 'NeuralNet with 3 Layers (4 relu, 2 relu, 2 relu nodes each) and Adam optimizer and increasing dropout layers',
            'Type': 'NeuralNet',
            'Layers': [
                {
                    'HiddenLayer': 1,
                    'NumberOfNodes': 4,
                    'Dropout': 0.1,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'HiddenLayer': 2,
                    'NumberOfNodes': 2,
                    'Dropout': 0.2,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'HiddenLayer': 2,
                    'NumberOfNodes': 2,
                    'Dropout': 0.3,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'NumberOfNodes': 1,
                    'ActivationFunction': tf.nn.sigmoid
                }
            ],
            'Optimizer': 'adam',
            'Loss': 'binary_crossentropy',
            'Metrics': ['accuracy'],
            'Epochs': 10
        },
        {
            'Name': 'NeuralNet with 3 Layers (4 relu, 2 relu, 2 relu nodes each) and NADAM optimizer and increasing dropout layers',
            'Type': 'NeuralNet',
            'Layers': [
                {
                    'HiddenLayer': 1,
                    'NumberOfNodes': 4,
                    'Dropout': 0.1,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'HiddenLayer': 2,
                    'NumberOfNodes': 2,
                    'Dropout': 0.2,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'HiddenLayer': 3,
                    'NumberOfNodes': 2,
                    'Dropout': 0.3,
                    'ActivationFunction': tf.nn.relu
                },
                {
                    'NumberOfNodes': 1,
                    'ActivationFunction': tf.nn.sigmoid
                }
            ],
            'Optimizer': 'nadam',
            'Loss': 'binary_crossentropy',
            'Metrics': ['accuracy'],
            'Epochs': 10
        },
        # {
        #     'Name': 'NeuralNet with 10 Layers (4 or 2 relu per layer) and NADAM optimizer and increasing dropout layers',
        #     'Type': 'NeuralNet',
        #     'Layers': [
        #         {
        #             'HiddenLayer': 1,
        #             'NumberOfNodes': 4,
        #             'Dropout': 0.1,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 2,
        #             'NumberOfNodes': 2,
        #             'Dropout': 0.2,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 3,
        #             'NumberOfNodes': 4,
        #             'Dropout': 0.3,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 4,
        #             'NumberOfNodes': 2,
        #             'Dropout': 0.4,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 5,
        #             'NumberOfNodes': 4,
        #             'Dropout': 0.5,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 6,
        #             'NumberOfNodes': 2,
        #             'Dropout': 0.6,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 7,
        #             'NumberOfNodes': 4,
        #             'Dropout': 0.7,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 8,
        #             'NumberOfNodes': 2,
        #             'Dropout': 0.8,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 9,
        #             'NumberOfNodes': 4,
        #             'Dropout': 0.9,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 10,
        #             'NumberOfNodes': 2,
        #             'Dropout': 0.9,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'NumberOfNodes': 1,
        #             'Dropout': 0,
        #             'ActivationFunction': tf.nn.sigmoid
        #         }
        #     ],
        #     'Optimizer': 'nadam',
        #     'Loss': 'binary_crossentropy',
        #     'Metrics': ['accuracy'],
        #     'Epochs': 10
        # },
        # {
        #     'Name': 'NeuralNet with 10 Layers (4 relu per layer) and NADAM optimizer and increasing dropout layers',
        #     'Type': 'NeuralNet',
        #     'Layers': [
        #         {
        #             'HiddenLayer': 1,
        #             'NumberOfNodes': 4,
        #             'Dropout': 0.1,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 2,
        #             'NumberOfNodes': 4,
        #             'Dropout': 0.2,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 3,
        #             'NumberOfNodes': 4,
        #             'Dropout': 0.3,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 4,
        #             'NumberOfNodes': 4,
        #             'Dropout': 0.4,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 5,
        #             'NumberOfNodes': 4,
        #             'Dropout': 0.5,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 6,
        #             'NumberOfNodes': 4,
        #             'Dropout': 0.6,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 7,
        #             'NumberOfNodes': 4,
        #             'Dropout': 0.7,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 8,
        #             'NumberOfNodes': 4,
        #             'Dropout': 0.8,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 9,
        #             'NumberOfNodes': 4,
        #             'Dropout': 0.9,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'HiddenLayer': 10,
        #             'NumberOfNodes': 4,
        #             'Dropout': 0.9,
        #             'ActivationFunction': tf.nn.relu
        #         },
        #         {
        #             'NumberOfNodes': 1,
        #             'Dropout': 0,
        #             'ActivationFunction': tf.nn.sigmoid
        #         }
        #     ],
        #     'Optimizer': 'nadam',
        #     'Loss': 'binary_crossentropy',
        #     'Metrics': ['accuracy'],
        #     'Epochs': 10
        # }
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
            if 'Dropout' in layer:
                # Add a dropout layer
                model.add(keras.layers.Dropout(layer['Dropout']))

        model.compile(optimizer=meta_model['Optimizer'],
                    loss=meta_model['Loss'],
                    metrics=meta_model['Metrics'])

    return model
