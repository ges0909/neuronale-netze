# [Deep Learning with Python, TensorFlow, and Keras tutorial](https://yewtu.be/watch?v=wQ8BIBpya2k&listen=false)
# [MNIST](https://paperswithcode.com/dataset/mnist)

# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist  # 28x28 images of handwritten digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"train={len(x_train)}, test={len(x_test)}")  # 60.000, 10.000

# plt.imshow(x_train[0])
# plt.show()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

val_loss, val_acc = model.evaluate(x_test, y_test)
print('Test loss:', val_loss)
print('Test accuracy:', val_acc)

model.save('models/number_reader.keras')
new_model = tf.keras.models.load_model('models/number_reader.keras')

predictions = new_model.predict(x_test)
print('Prediction:', np.argmax(predictions[0]))
