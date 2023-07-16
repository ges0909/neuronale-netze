# [In 5min zum ersten neuronalen Netz in Python](https://yewtu.be/watch?v=13QViOVy9Tg)

from keras import *

# from keras import __version__

# (__version__)

# neuronales Netz erzeugen
model = models.Sequential()

# input layer
# - 'Dense' = jedes Neuron mit jedem Neuron der nächsten Schicht verknüpfen
# - 'units' = Anzahl der Neuronen
# - 'input_shape' = Dimension der Eingangsdaten, '1' = eindimensional = Listen
model.add(layers.Dense(units=3, input_shape=[1]))

# hidden layer
model.add(layers.Dense(units=2))

# output layer
model.add(layers.Dense(units=1))

# training
# - 'mean_squared_error' = minimale Abweichung zum Quadrat
# - "loss"-Funktion bestimmt, wann eine Abweichung vom gewünschten Ergebnis "gut" oder "schlecht "ist
eingang = [1, 2, 3, 4, 5]
ausgang = [10, 20, 30, 40, 50]
model.compile(loss='mean_squared_error', optimizer='adam')

# 5000 Trainingsdurchläufe
model.fit(x=eingang, y=ausgang, epochs=5000)

# Prüfen der Lernerfolge
print(model.predict([6]))
print(model.predict([8]))
print(model.predict([10]))
