from tensorflow.keras.datasets.mnist  import load_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
import numpy as np


# ------ Load data
def load_data():
    return mnist.load_data()

# ------ Preprocess data
def preprocess_data(trainX, testX, trainY, testY):
    # Normalize
    trainX = trainX.astype('float32') / 255.0
    testX = testX.astype('float32') / 255.0

    # One-hot encode labels
    trainY_cat = to_categorical(trainY, 10)
    testY_cat = to_categorical(testY, 10)

    return trainX, testX, trainY, testY, trainY_cat, testY_cat

# ------ Build model
def build_model():
    inputs = Input((28, 28))
    x = Flatten()(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ------ Train model
def train_model(model, trainX, trainY_cat):
    return model.fit(trainX, trainY_cat,
                     epochs=20,
                     batch_size=32,
                     validation_split=0.1)

# ------ Evaluate model
def evaluate_model(model, testX, testY_cat):
    loss, accuracy = model.evaluate(testX, testY_cat)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    return loss, accuracy

# ------ Predict and compare
def predict_and_compare(model, testX, testY):
    predictions = model.predict(testX)
    predicted_labels = np.argmax(predictions, axis=1)

    print("\nFirst 10 predictions vs actual labels:")
    for i in range(10):
        print(f"Predicted: {predicted_labels[i]}, Actual: {testY[i]}")

# ------ Main
def main():
    # Load
    (trainX, trainY), (testX, testY) = load_data()
    print("Train:", trainX.shape, trainY.shape)
    print("Test:", testX.shape, testY.shape)

    # Preprocess
    trainX, testX, trainY, testY, trainY_cat, testY_cat = preprocess_data(trainX, testX, trainY, testY)

    # Build
    model = build_model()

    # Train
    train_model(model, trainX, trainY_cat)

    # Evaluate
    evaluate_model(model, testX, testY_cat)

    # Predict
    predict_and_compare(model, testX, testY)

if __name__ == "__main__":
    main()


import matplotlib.pyplot as plt
from tensorflow.keras.datasets.mnist import load_data

# Load MNIST dataset
(trainX, trainY), (testX, testY) = load_data()

# Plot the first 10 images in the training set
plt.figure(figsize=(12, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(trainX[i], cmap='gray')
    plt.title(f"{trainY[i]}")
    plt.axis('off')
plt.suptitle("Sample MNIST Images with Labels")
plt.show()
