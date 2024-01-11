import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar tu modelo entrenado
modelo = load_model('mi_modelo.h5')  # Reemplaza 'modelo_colores.h5' con el nombre real de tu modelo

# Cargar la imagen de prueba en color (sin escala de grises)
imagen_prueba = cv2.imread('imagen_prueba.jpg')  # Cargar la imagen en su formato original en color

# Redimensionar la imagen de prueba para que coincida con las dimensiones esperadas por el modelo
alto_deseado = 50
ancho_deseado = 50
imagen_prueba_redimensionada = cv2.resize(imagen_prueba, (ancho_deseado, alto_deseado))

# Normalizar la imagen
imagen_prueba_redimensionada = imagen_prueba_redimensionada / 255.0

# Asegurarse de que la imagen tenga la misma forma que las imágenes de entrenamiento
imagen_prueba_redimensionada = np.expand_dims(imagen_prueba_redimensionada, axis=0)  # Agregar dimensión para coincidir con el formato de entrada del modelo

# Realizar la predicción con la imagen redimensionada
prediccion = modelo.predict(imagen_prueba_redimensionada)

# Diccionario de nombres de colores asociados a los índices de clase
nombres_colores = {
    0: 'Azul',
    1: 'Amarillo Limon',
    2: 'Rojo',
    3: 'Verde',
    4: 'Ocre'
}

# Obtener el índice de la clase predicha
indice_predicho = np.argmax(prediccion)

# Obtener el nombre del color correspondiente al índice predicho
color_predicho = nombres_colores[indice_predicho]

# Mostrar la cantidad asociada a cada clase
print(f"Es el color {color_predicho} y requiere de:")
for indice, nombre_color in nombres_colores.items():
    cantidad = int(prediccion[0][indice] * 100)  # Multiplicar por 100 para obtener un número entero
    print(f"{nombre_color} ({cantidad}%)")

