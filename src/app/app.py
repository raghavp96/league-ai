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


def __runSpecific(modelName="", method="fixed", onlyUseBasicFeatures=False, epochs=1):
    # Get training data/labels, test data/labels, development data/labels from data as numpy arrays
    ((train_data, train_labels), (test_data, test_labels), (dev_data, dev_labels)) = data.get_data(preparation_method=method, onlyUseBasicFeatures=onlyUseBasicFeatures)

    # Get the models
    model_metadata = models.get_models()

    the_model = None
    for model in model_metadata:
        if model["Name"] == modelName:
           the_model = model
           break
    
    if the_model is not None:
        the_model["Model"].fit(train_data, train_labels, epochs=epochs)

    # Run on test set
    test_loss, test_acc = the_model["Model"].evaluate(test_data, test_labels) 


if __name__ == "__main__":
    method_mentioned = "fixed"
    onlyUseBasicFeatures = False
    modelName = ""
    epochs = 1
    for arg in sys.argv[1:]:
        if arg in ["fixed", "random"]:
            method_mentioned = arg
        elif arg in "onlyBasic":
            onlyUseBasicFeatures = True
        else:
            try :
                epochs = int(arg)
            except ValueError:
                modelName = arg
    
    if modelName != "":
        __runSpecific(modelName=modelName, method=method_mentioned, onlyUseBasicFeatures=onlyUseBasicFeatures, epochs=epochs)
    else:
        print("Run mode: " + method_mentioned + " with onlyBasicFeatures: " + str(onlyUseBasicFeatures))
        __run(method_mentioned, onlyUseBasicFeatures)
        