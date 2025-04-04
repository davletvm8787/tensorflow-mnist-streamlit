import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

MODEL_PATH = "mnist_model.h5"

def train_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=6, validation_split=0.1)
    model.evaluate(x_test, y_test, verbose=2)
    model.save(MODEL_PATH)
    print(f"✅ Модель сохранена как {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
