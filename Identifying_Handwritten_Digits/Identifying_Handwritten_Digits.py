import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255
x_test = x_test.reshape(-1, 784).astype("float32") / 255

inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation='relu', name='first_layer')(inputs)
x = layers.Dense(256, activation='relu', name='second_layer')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=2)
model.evaluate(x_test, y_test, batch_size=128, verbose=2)
