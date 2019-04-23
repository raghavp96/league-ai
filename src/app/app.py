import data
import models
import sys
import results as rs_write


def __run(method, onlyUseBasicFeatures):
    # Get training data/labels, test data/labels, development data/labels from data as numpy arrays
    ((train_data, train_labels), (test_data, test_labels), (dev_data, dev_labels)) = data.get_data(preparation_method=method, onlyUseBasicFeatures=onlyUseBasicFeatures)

    epochs = [10, 50, 100]
    results = {}

    for epoch in epochs:
        # Get the models
        model_metadata = models.get_models()

        # Train each model
        for model in model_metadata:
            model["Model"].fit(train_data, train_labels, epochs=epoch)

        # Test the model with development data
        for model in model_metadata:
            test_loss, test_acc = model["Model"].evaluate(dev_data, dev_labels)

            resultForThisEpoch = {
                "Epoch" : epoch,
                "Accuracy" : test_acc,
                "Loss" : test_loss
            }

            if model["Name"] not in results:
                results[model["Name"]] = [resultForThisEpoch]
            else:
                results[model["Name"]].append(resultForThisEpoch)

        # Run the model on the test data
        # predictions = best_model.predict(test_data)
        # for i in range(len(predictions)):
        #     print("Predicted: " + predictions[i] + " Actual: " + test_labels[i])
    rs_write.writeDict(results, epochs=epochs)


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
        