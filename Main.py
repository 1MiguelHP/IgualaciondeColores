import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from pyqt5_plugins.examplebutton import QtWidgets
import cv2
from tensorflow.keras.models import load_model
from vent import Ui_Main
from resultados import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):

        super(MainWindow, self).__init__()
        self.imagen_cargada = None
        self.res = QtWidgets.QMainWindow()
        self.ui = Ui_Main()
        self.ui.setupUi(self)
        self.ou = Ui_MainWindow()
        self.ou.setupUi(self.res)
        self.setAcceptDrops(True)
        self.ui.cargar_imagen.clicked.connect(self.Cargar_Imagen_Manual)
        self.ui.pushButton.clicked.connect(self.Identificar_Color)

        self.ui.label.dragEnterEvent = self.dragEnterEvent
        self.ui.label.dropEvent = self.dropEvent
        self.ui.label.setStyleSheet('''
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
            }
        ''')

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        image_path = event.mimeData().urls()[0].toLocalFile()
        print("Ruta de la imagen:", image_path)  # Verifica si la ruta se muestra correctamente
        pixmap = QPixmap(image_path)
        self.ui.label.setPixmap(pixmap)
        self.imagen_cargada = image_path  # Almacena la ruta del archivo

    def Cargar_Imagen_Manual(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        filename, _ = file_dialog.getOpenFileName(self, "Seleccionar imagen", "",
                                                  "Images (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)

        if filename:
            pixmap = QPixmap(filename)
            self.ui.label.setPixmap(pixmap)
            self.imagen_cargada = pixmap.toImage()

    def Identificar_Color(self):

        # Cargar tu modelo entrenado
        modelo = load_model('mi_modelo.h5')  # Reemplaza 'modelo_colores.h5' con el nombre real de tu modelo

        # Cargar la imagen de prueba en color (sin escala de grises)
        imagen_prueba = cv2.imread(self.imagen_cargada)  # Cargar la imagen en su formato original en color

        # Redimensionar la imagen de prueba para que coincida con las dimensiones esperadas por el modelo
        alto_deseado = 50
        ancho_deseado = 50
        imagen_prueba_redimensionada = cv2.resize(imagen_prueba, (ancho_deseado, alto_deseado))

        # Normalizar la imagen
        imagen_prueba_redimensionada = imagen_prueba_redimensionada / 255.0

        # Asegurarse de que la imagen tenga la misma forma que las imágenes de entrenamiento
        imagen_prueba_redimensionada = np.expand_dims(imagen_prueba_redimensionada,
                                                      axis=0)  # Agregar dimensión para coincidir con el formato de entrada del modelo

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
        texto = ""  # Variable para almacenar todos los textos de colores y porcentajes

        for indice, nombre_color in nombres_colores.items():
            cantidad = int(prediccion[0][indice] * 100)  # Multiplicar por 100 para obtener un número entero
            texto += f"{nombre_color} ({cantidad}%)\n"  # Agregar texto de color y porcentaje a la variable 'texto'
            print(texto)
        # Establecer el texto acumulado en el QTextEdit
        self.ou.textEdit.setPlainText(texto)
        self.res.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


