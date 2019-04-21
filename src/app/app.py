import data
import models
import sys


def __run(method):
    # Get training data/labels, test data/labels from data as numpy arrays
    ((train_data, train_labels), (test_data, test_labels)) = data.get_data(method)

    # Get the models
    model_metadata = models.get_models()

    # Train each model
    for model in model_metadata:
        model["Model"].fit(train_data, train_labels, epochs=10)

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

    # model_weights_per_layer = []
    # for model in model_metadata:
    #     model_weights = {}
    #     model_weights["Name"] = model["Name"]
    #     model_weights["Layers"] = {}
    #     for (idx, layer) in model["Model"].layers:
    #         model_weights["Layers"]["idx"] = layer.get_weights()

    # print(model_weights_per_layer)

    print("Best Model's Results - Model Name: " + max_name + ", Accuracy: " + str(max_acc) + ", Loss: " + str(max_loss))
    print("Best Model Details: " + str([model["Model"].summary() for model in model_metadata if model["Name"] == max_name]) )


if __name__ == "__main__":
    method_mentioned = False
    for arg in sys.argv[1:]:
        if not method_mentioned:
            if arg in "fixed" or "random":
                method_mentioned = True
                __run(arg)
    if not method_mentioned:
        __run("random")
        