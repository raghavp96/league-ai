import data
# import numpy
import models

# Get training data/labels, test data/labels from data as numpy arrays
((train_data, train_labels), (test_data, test_labels)) = data.get_data()

# Get the models
ml_models = models.get_models()

# Train each model
print("Training all models")
for model in ml_models:
    print("Training the model")
    model.fit(train_data, train_labels, epochs=10)
    print("Trained")
print("Training completed.")

# Test the model
print("Testing all models")
for model in ml_models:
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print('Model Test accuracy:', test_acc)
print("All models tested.")

# predictions = model.predict(test_data)

# print(numpy.argmax(predictions[0]))
# print(test_labels[0])
