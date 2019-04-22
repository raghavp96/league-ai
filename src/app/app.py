import data
import models
import sys


def __run(method, onlyUseBasicFeatures):
    # Get training data/labels, test data/labels from data as numpy arrays
    ((train_data, train_labels), (test_data, test_labels)) = data.get_data(preparation_method=method, onlyUseBasicFeatures=onlyUseBasicFeatures)

    # Get the models
    model_metadata = models.get_models()

    # Train each model
    for model in model_metadata:
        model["Model"].fit(train_data, train_labels, epochs=model["Epochs"])

    max_acc = float("-inf")
    max_name = ""
    max_loss = float("-inf")

    # Test the model
    for model in model_metadata:
        test_loss, test_acc = model["Model"].evaluate(test_data, test_labels)
        if (max_acc < test_acc):
            max_acc = test_acc
            max_loss = test_loss
            max_name = model['Name']

    print("Best Model's Results - Model Name: " + max_name + ", Accuracy: " + str(max_acc) + ", Loss: " + str(max_loss))
    print("Best Model Details: " + str([model["Model"].summary() for model in model_metadata if model["Name"] == max_name]) )


if __name__ == "__main__":
    method_mentioned = "fixed"
    onlyUseBasicFeatures = False
    for arg in sys.argv[1:]:
        if arg in ["fixed", "random"]:
            method_mentioned = arg
        elif arg in "onlyBasic":
            onlyUseBasicFeatures = True

    print("Run mode: " + method_mentioned + " with onlyBasicFeatures: " + str(onlyUseBasicFeatures))
    __run(method_mentioned, onlyUseBasicFeatures)
        