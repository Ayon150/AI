import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score, mean_absolute_error

# Define the function
def f(x):
    return 5*x**2 + 10*x - 2

# Generate dataset
X = np.linspace(-10, 10, 5000).reshape(-1, 1)
y = f(X)

# Split into train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build FCFNN model
model = keras.Sequential([
    layers.Dense(32, activation="relu", input_shape=(1,)),
    layers.Dense(64, activation="relu"),

    layers.Dense(128),
    layers.LeakyReLU(alpha=0.01),

    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

# Compile
model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")

# Train
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=1000, batch_size=32, verbose=0)

# Predict
y_pred = model.predict(X_test)

# --- Evaluate ---
mse = model.evaluate(X_test, y_test, verbose=0)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"R² Score (Accuracy): {r2:.4f}")

# --- Plot training loss ---
plt.figure(figsize=(7,5))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid()
plt.savefig("training_loss.png", dpi=150)
plt.show()

# --- Plot function approximation ---
plt.figure(figsize=(7,5))
plt.plot(X, y, label="Original f(x)", linewidth=2)
plt.plot(X, model.predict(X), label="Predicted f(x)", linestyle="dashed")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Function Approximation with FCFNN")
plt.legend()
plt.grid()
plt.savefig("function_comparison.png", dpi=150)
plt.show()
