# [Künstliches neuronales Netz - Python programmieren](https://yewtu.be/watch?v=d3-j-hq5AD8)

import pandas as pd
from keras.layers import Dense, Input
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Daten aus .csv-Datei einlesen
df = pd.read_csv('data/housepricedata.csv')

# Input-Daten aus Spalten 0..9 erzeugen
x = df.iloc[:, 0:10]
# print(x)

# Output-Daten aus Spalte 10 erzeugen
y = df.iloc[:, 10]
# print(y)

# Input auf Werte 0..1 skalieren
x = preprocessing.MinMaxScaler().fit_transform(x)

# Train-Test-Split anlegen
# - 70 Prozent der Daten für das Training verwenden
# - 30 Prozent der Daten für den Test verwenden
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3)

# Neuronales Netz anlegen
model = Sequential()

# input layer
model.add(Input(shape=(10,)))

# hidden layers
# - 'Dense' = jedes Neuron mit jedem Neuron der nächsten Schicht verknüpfen
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=32, activation="relu"))

# output layer
model.add(Dense(units=1, activation="sigmoid"))

# training
# - "loss"-Funktion bestimmt, wann eine Abweichung vom gewünschten Ergebnis "gut" oder "schlecht "ist
# - 'mean_squared_error' = minimale Abweichung zum Quadrat
model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy', )

# 10 Trainingsdurchläufe
model.fit(x=x_train, y=y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val), verbose=1, )

#
test = [
    [8450, 7, 5, 5856, 2, 1, 3, 8, 0, 548],  # 1
    [14260, 8, 5, 1145, 2, 1, 4, 9, 1, 548],  # 1
    [6120, 7, 8, 832, 1, 0, 2, 5, 0, 576],  # 0
    [12968, 5, 6, 912, 1, 0, 2, 4, 0, 352],  # 0
]

test = preprocessing.MinMaxScaler().fit_transform(test)

print(model.predict(test))

model.save('models/house-price.keras')
