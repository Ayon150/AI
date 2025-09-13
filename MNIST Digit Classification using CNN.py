from tensorflow.keras.datasets.mnist  import load_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
# ------ Load data
def load_data():
    from tensorflow.keras.datasets.mnist import load_data as mnist_load_data
    return mnist_load_data()
# ------ Preprocess data
def preprocess_data(trainX, testX, trainY, testY):
    trainX = trainX.astype('float32') / 255.0
    testX = testX.astype('float32') / 255.0

    # Reshape to (28, 28, 1) for CNN
    trainX = trainX.reshape(-1, 28, 28, 1)
    testX = testX.reshape(-1, 28, 28, 1)

    # One-hot encode labels
    trainY_cat = to_categorical(trainY, 10)
    testY_cat = to_categorical(testY, 10)

    return trainX, testX, trainY, testY, trainY_cat, testY_cat
# ------ Build model
def build_model():
    inputs = Input(shape=(28, 28, 1))

    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
# ------ Train model
def train_model(model, trainX, trainY_cat):
    return model.fit(trainX, trainY_cat,
                     epochs=10,
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
