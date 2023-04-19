import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import pyplot as plt

numeros_prueba = range(0,10)
imagenes = []
etiquetas_imagenes = []

num = 0
guardar = str()

#0.28 0.28 0,1       0.28 29.56 0,2
for number in numeros_prueba:
    imgReaded= f'grilla{str(number)}.jpg'
    img = cv2.imread(imgReaded, cv2.IMREAD_GRAYSCALE)
    for i in range(0,280*11,28):
        ini_ver = i 
        fin_ver = i+27
        for j in range (0,280,28):
            ini_hor = j
            fin_hor = j+27
            #num+=1
            #guardar = str(num)+".jpg"
            img_numero = img[ini_ver:fin_ver, ini_hor:fin_hor].copy()
            img_numero = cv2.resize(img_numero, (28, 28))
            img_numero = cv2.bitwise_not(img_numero)
            imagenes.append(img_numero)
            etiquetas_imagenes.append(number)
            #cv2.imwrite(guardar ,img[ini_ver:fin_ver, ini_hor:fin_hor])

imagenes = np.array(imagenes)
etiquetas_imagenes = np.array(etiquetas_imagenes)

modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # Capa de entrada que aplana las imágenes de 28x28 píxeles a un vector de 784 elementos
    tf.keras.layers.Dense(1545, activation='relu'), # Capa oculta con 128 neuronas y función de activación ReLU
   



    #tf.keras.layers.Dense(128*5, activation='relu'), # Capa oculta con 128 neuronas y función de activación ReLU


    


    tf.keras.layers.Dense(10) # Capa de salida con 10 neuronas (una por cada número)
])

# Compilar el modelo con una función de pérdida y un optimizador
modelo.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Ajustar el modelo a las imágenes de entrenamiento
modelo.fit(imagenes, etiquetas_imagenes, epochs=25)

resultado = modelo.evaluate(imagenes, etiquetas_imagenes)
print('Pérdida en el conjunto de prueba:', resultado[0])
print('Precisión en el conjunto de prueba:', resultado[1])

print('Real: 1')
img_nueva = cv2.imread("1prueba.jpg", 0)

# Redimensionar la imagen a 28x28 píxeles
img_nueva = cv2.resize(img_nueva, (28, 28))
img_nueva = cv2.bitwise_not(img_nueva)


# Invertir los colores de la imagen
#img_nueva = cv2.bitwise_not(img_nueva)

# convertir la imagen a un tensor y normalizar los valores de píxeles
#img_prueba = tf.keras.utils.normalize(img_nueva.reshape((1, 28, 28, 1)), axis=1)

# Escalar los valores de los píxeles a un rango de 0 a 1
img_nueva = img_nueva.astype('float32') / 255

# Aplanar la imagen en un vector de 784 elementos
img_nueva = img_nueva.reshape(1, 28, 28)



# Hacer una predicción sobre la imagen nueva
prediccion = modelo.predict(img_nueva)

# Obtener el número predicho como el índice con la mayor probabilidad
numero_predicho = np.argmax(prediccion)

print('El número predicho es:', numero_predicho)


print('Real: 3')

img_nueva = cv2.imread("3-2prueba.jpg", cv2.IMREAD_GRAYSCALE)
img_nueva = cv2.bitwise_not(img_nueva)

# Redimensionar la imagen a 28x28 píxeles
img_nueva = cv2.resize(img_nueva, (28, 28))

# Invertir los colores de la imagen
#img_nueva = cv2.bitwise_not(img_nueva)

# Escalar los valores de los píxeles a un rango de 0 a 1
img_nueva = img_nueva.astype('float32') / 255.0

# Aplanar la imagen en un vector de 784 elementos
img_nueva = img_nueva.reshape(1, 784)

# Hacer una predicción sobre la imagen nueva
prediccion = modelo.predict(img_nueva)

# Obtener el número predicho como el índice con la mayor probabilidad
numero_predicho = np.argmax(prediccion)

print('El número predicho es:', numero_predicho)


print('Real: 8')

img_nueva = cv2.imread("8-1prueba.jpg", cv2.IMREAD_GRAYSCALE)
img_nueva = cv2.bitwise_not(img_nueva)

# Redimensionar la imagen a 28x28 píxeles
img_nueva = cv2.resize(img_nueva, (28, 28))

# Invertir los colores de la imagen
#img_nueva = cv2.bitwise_not(img_nueva)

# Escalar los valores de los píxeles a un rango de 0 a 1
img_nueva = img_nueva.astype('float32') / 255.0

# Aplanar la imagen en un vector de 784 elementos
img_nueva = img_nueva.reshape(1, 784)

# Hacer una predicción sobre la imagen nueva
prediccion = modelo.predict(img_nueva)

# Obtener el número predicho como el índice con la mayor probabilidad
numero_predicho = np.argmax(prediccion)

print('El número predicho es:', numero_predicho)


print('Real: 6')

img_nueva = cv2.imread("6prueba.jpg", cv2.IMREAD_GRAYSCALE)
img_nueva = cv2.bitwise_not(img_nueva)

# Redimensionar la imagen a 28x28 píxeles
img_nueva = cv2.resize(img_nueva, (28, 28))

# Invertir los colores de la imagen
#img_nueva = cv2.bitwise_not(img_nueva)

# Escalar los valores de los píxeles a un rango de 0 a 1
img_nueva = img_nueva.astype('float32') / 255.0

# Aplanar la imagen en un vector de 784 elementos
img_nueva = img_nueva.reshape(1, 784)

# Hacer una predicción sobre la imagen nueva
prediccion = modelo.predict(img_nueva)

# Obtener el número predicho como el índice con la mayor probabilidad
numero_predicho = np.argmax(prediccion)

print('El número predicho es:', numero_predicho)
