#file upload section
from google.colab import files
uploaded = files.upload()

#import section
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load DATASET
# -----------------------------
def load_data(path="/content/my_dataset.npz"):
  data = np.load(path)
  X_train = data['x_train']
  X_test = data['x_test']
  y_train = data['y_train']
  y_test = data['y_test']
  print("Train:", X_train.shape, " Test:", X_test.shape)
  return X_train, X_test, y_train, y_test

# Normalize dataset
# -----------------------------
def normalize_data(X_train, X_test):
  X_train = X_train.astype("float32") / 255.0
  X_test = X_test.astype("float32") / 255.0
  return X_train, X_test

def build_model():
    inputs = Input((28, 28))
    x = Flatten()(inputs)

    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)

    outputs = Dense(10, activation="softmax")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)

    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, lr_schedule],
        verbose=2
    )
    return history

#  Plot Training
# -----------------------------
def plot_history(history):
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Main Execution
# -----------------------------
X_train, X_test, y_train, y_test = load_data()
X_train, X_test = normalize_data(X_train, X_test)
model = build_model()
history = train_model(model, X_train, y_train, X_test, y_test)
plot_history(history)

# Final evaluation
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Accuracy: {acc*100:.2f}%")

 colab link : https://colab.research.google.com/drive/1lT27__vjg0buQ3i6EnPCoaZky7HosPJg?usp=sharing
