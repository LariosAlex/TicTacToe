import cv2
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

model = tf.keras.models.load_model('modelo_entrenado2.h5')


def detect_squares(frame):
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Aplicar desenfoque gaussiano
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detectar bordes con Canny
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    # Encontrar contornos en los bordes
    squares, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return squares

def filter_largest_square(squares):
    largest_square = None
    max_area = 0

    if squares is not None:
        for square in squares:
            epsilon = 0.01 * cv2.arcLength(square, True)
            approx = cv2.approxPolyDP(square, epsilon, True)
            if len(approx) == 4:  # Cuadrado
                area = cv2.contourArea(square)
                if area > max_area and area > 50000:  # Ajustar área al cuadrado
                    max_area = area
                    largest_square = approx

    return largest_square

def detect_small_squares(roi):
    # Convertir a escala de grises
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Aplicar desenfoque gaussiano
    blur_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    # Detectar bordes con Canny
    edges_roi = cv2.Canny(blur_roi, 50, 150, apertureSize=3)
    # Encontrar contornos en los bordes
    small_squares, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return small_squares

def filter_small_squares(small_squares):
    filtered_squares = []
    min_area = 100  # Ajusta el área mínima para considerar un cuadrado pequeño

    if small_squares is not None:
        for small_square in small_squares:
            epsilon = 0.01 * cv2.arcLength(small_square, True)
            approx = cv2.approxPolyDP(small_square, epsilon, True)
            if len(approx) == 4:  # Cuadrado
                area = cv2.contourArea(small_square)
                if area > min_area:
                    filtered_squares.append(approx)

    return filtered_squares

def draw_squares(frame, squares, color):
    for square in squares:
        cv2.drawContours(frame, [square], 0, color, 2)

def draw_square(frame, square):
    if square is not None:
        cv2.drawContours(frame, [square], 0, (255, 0, 0), 2)

def predict_content(image):

    preprocessed_image = preprocess_image(image)

    if preprocessed_image is not None:
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions)
        return predicted_class
    else:
        return -1 


def preprocess_image(image):
    if image is None:
        return None
    
    if len(image.shape) == 2:  # Si la imagen es en escala de grises
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:  # Si la imagen tiene un solo canal (escala de grises)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Cambiar el tamaño de la imagen al tamaño esperado por el modelo (150x150)
    resized_image = cv2.resize(image, (150, 150))

    # Aplicar desenfoque gaussiano
    blur_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    
    # Normalizar los valores de píxeles en el rango [0, 1]
    preprocessed_image = blur_image / 255.0
    
    # Agregar una dimensión extra para adaptarse al formato de entrada del modelo
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    return preprocessed_image
def calcularJugada(arrayTablero):
    print(arrayTablero)

def main():
    # Capturar video desde la cámara (puedes cambiar el número 0 si tienes varias cámaras)
    cap = cv2.VideoCapture(0)

    while True:
        # Leer un frame del video
        ret, frame = cap.read()

        # Inicializar filtered_small_squares incluso si largest_square es None
        filtered_small_squares = []

        # Procesar la imagen
        squares = detect_squares(frame)
        largest_square = filter_largest_square(squares)

        if largest_square is not None:
            x, y, w, h = cv2.boundingRect(largest_square)
            roi = frame[y:y + h, x:x + w]

            # Detectar cuadrados pequeños dentro del ROI
            small_squares = detect_small_squares(roi)
            filtered_small_squares = filter_small_squares(small_squares)

           # Para cada cuadrado pequeño, hacer una predicción y colocar la etiqueta
        arrayJugada = []
        for idx, small_square in enumerate(filtered_small_squares):
            jugadaActual = []
            x_s, y_s, w_s, h_s = cv2.boundingRect(small_square)
            small_roi = roi[y_s:y_s + h_s, x_s:x_s + w_s]

            # Hacer la predicción del contenido del cuadrado pequeño
            predicted_class = predict_content(small_roi)

            # Asignar etiqueta según la predicción
            if predicted_class == 1:
                label = "Equis"
                jugadaActual.append(predicted_class)
            elif predicted_class == 0:
                jugadaActual.append(predicted_class)
                label = "Circulo"
            else:
                jugadaActual.append(2)
                label = "Vacio"

            # Ajustar la posición para centrar el texto
            text_x = x_s + w_s // 2 - 30
            text_y = y_s + h_s // 2

           # Obtener las dimensiones del texto
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Crear un fondo blanco debajo del texto
            cv2.rectangle(roi, (text_x, text_y - label_height), (text_x + label_width, text_y + baseline), (255, 255, 255), -1)

            # Dibujar etiqueta sobre el cuadrado pequeño
            cv2.putText(roi, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)



            # Dibujar cuadrados pequeños en rojo
            draw_squares(roi, filtered_small_squares, (0, 0, 255))

            # Dibujar el cuadrado más grande en azul
            draw_square(frame, largest_square)

            cv2.imshow("ROI", roi)
            
        if arrayJugada != jugadaActual:
            arrayJugada = jugadaActual
            calcularJugada(arrayJugada)
            
        # Mostrar la imagen original
        cv2.imshow("Original", frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la captura de video y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()