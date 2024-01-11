from Main import MainWindow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2


# Cargar el archivo CSV
data = pd.read_csv('colores.csv')  # Asegúrate de que el archivo esté en la misma carpeta que este script o proporciona la ruta correcta

# Cargar las rutas de las imágenes y convertirlas a una lista
rutas_imagenes = data['imagen'].values.tolist()

# Cargar las imágenes desde las rutas a color (sin escala de grises)
imagenes = []
ancho_deseado = 50  # Ajusta estos valores según tus necesidades
alto_deseado = 50
for ruta in rutas_imagenes:
    imagen = cv2.imread(ruta)  # No usar cv2.IMREAD_GRAYSCALE para cargar a color
    imagen = cv2.resize(imagen, (ancho_deseado, alto_deseado))  # Ajusta ancho_deseado y alto_deseado
    imagenes.append(imagen)

# Convertir a un arreglo numpy
imagenes = np.array(imagenes)

# Normalizar los valores de los píxeles (escala 0-1)
imagenes = imagenes / 255.0

# Etiquetas correspondientes a cada imagen
etiquetas = data[['Azul', 'Amarillo Limon', 'Rojo', 'Verde', 'Ocre']].values

# Dividir los datos en conjuntos de entrenamiento y prueba
imagenes_entrenamiento, imagenes_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(
    imagenes, etiquetas, test_size=0.2, random_state=42
)

# Crear modelo de red neuronal convolucional para imágenes a color (RGB)
modelo = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(alto_deseado, ancho_deseado, 3)),  # Cambio en input_shape a 3 canales (RGB)
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compilar el modelo
modelo.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(imagenes_entrenamiento, etiquetas_entrenamiento, epochs=20, batch_size=32, validation_data=(imagenes_prueba, etiquetas_prueba))

# Guardar el modelo
nombre_archivo_modelo = 'mi_modelo.h5'  # Especifica el nombre que desees para el archivo del modelo
modelo.save(nombre_archivo_modelo)