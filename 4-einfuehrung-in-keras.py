# [Einführung in Keras](https://datasolut.com/einfuehrung-in-keras/)

# import von TF
import tensorflow as tf

# Handgeschriebene Ziffern laden
mnist = tf.keras.datasets.mnist

# Aufteilung in Training- und Testset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# TF Bilderkennungsmodell
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Crossentropy für die 10 Zahlen Klassen
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modell-Fitting und Evaluation
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
