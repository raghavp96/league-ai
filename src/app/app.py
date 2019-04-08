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

    print("Best Model's Results - Model Name: " + max_name + ", Accuracy: " + str(max_acc) + ", Loss: " + str(max_loss))

# predictions = model.predict(test_data)

# print(numpy.argmax(predictions[0]))
# print(test_labels[0])

if __name__ == "__main__":
    method_mentioned = False
    for arg in sys.argv[1:]:
        if not method_mentioned:
            if arg in "fixed" or "random":
                method_mentioned = True
                __run(arg)
    if not method_mentioned:
        __run("random")