import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train[0].shape

len(x_train)

plt.imshow(x_train[0], cmap = 'gray',  vmin=0, vmax=255)
y_train[0]

x_train = x_train / 255
x_test = x_test / 255
x_train[0]

x_train_flatten = x_train.reshape(len(x_train), 28*28)
x_test_flatten = x_test.reshape(len(x_test), 28*28)
x_train_flatten.shape 
x_train_flatten[0]

model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid' )
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy'),
model.fit(x_train_flatten, y_train, epochs=5)

model.evaluate(x_test_flatten, y_test)

y_predicted = model.predict(x_test_flatten)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.imshow(x_test[80], cmap='gray', vmin=0, vmax=1)
print(f'y_test: {y_test[80]}')

y_predicted = model.predict(x_test_flatten)
np.argmax(y_predicted[80])
print(y_test[80] == np.argmax(y_predicted[80]))
print(y_test[80])


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')  #100 capas ocultas y 10 de salidas.
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_flatten, y_train, epochs=5)
model.evaluate(x_test_flatten,y_test)


y_predicted = model.predict(x_test_flatten)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)                        

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test,y_test)