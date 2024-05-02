import cv2
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import math

#Hay que cambiar el modelo entrenado con los numero correctoss!!!
model = tf.keras.models.load_model('modelo_entrenado2.h5')

last_player = 1

jugador = 2

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

def calcularJugada(arrayTablero, last_player):
    jugador = 2
    if es_turno(arrayTablero, jugador, last_player):
        mejor_jugada = minmax(arrayTablero, jugador)
        arrayTablero[mejor_jugada] = 1
        print("Mejor jugada para el jugador", jugador, "es en la posición", mejor_jugada)
        mostrar_tablero(arrayTablero)
    else:
        print("No es el turno del jugador", jugador)

def es_turno(tablero, jugador, last_player):
    count_circulo = tablero.count(0)
    count_cruz = tablero.count(1)

    if count_circulo > count_cruz:
        return jugador == 1
    elif count_circulo < count_cruz:
        return jugador == 0
    else:
        return jugador != last_player 


def obtener_ganador(tablero):
    """
    Verifica si hay un ganador en el tablero.
    Retorna 0 si gana el círculo, 1 si gana la cruz, o None si hay empate o el juego no ha terminado.
    """
    for i in range(3):
        # Filas
        if tablero[i*3] == tablero[i*3 + 1] == tablero[i*3 + 2] and tablero[i*3] != 2:
            return tablero[i*3]
        # Columnas
        if tablero[i] == tablero[i + 3] == tablero[i + 6] and tablero[i] != 2:
            return tablero[i]
    
    # Diagonales
    if tablero[0] == tablero[4] == tablero[8] and tablero[0] != 2:
        return tablero[0]
    if tablero[2] == tablero[4] == tablero[6] and tablero[2] != 2:
        return tablero[2]
    
    # Si no hay ganador pero el tablero está lleno, hay empate
    if 2 not in tablero:
        return None
    
    # Si no hay ganador ni empate, el juego no ha terminado
    return None

def minmax(tablero, jugador):
    """
    Algoritmo Minimax para encontrar la mejor jugada posible.
    """
    def maximizar(tablero, jugador):
        ganador = obtener_ganador(tablero)
        if ganador is not None:
            if ganador == jugador:
                return 1
            elif ganador is None:
                return 0
            else:
                return -1

        mejor_valor = -math.inf
        for i in range(9):
            if tablero[i] == 2:
                tablero[i] = jugador
                valor = minimizar(tablero, jugador)
                tablero[i] = 2
                mejor_valor = max(mejor_valor, valor)
        return mejor_valor

    def minimizar(tablero, jugador):
        otro_jugador = 1 if jugador == 0 else 0
        ganador = obtener_ganador(tablero)
        if ganador is not None:
            if ganador == jugador:
                return 1
            elif ganador is None:
                return 0
            else:
                return -1

        mejor_valor = math.inf
        for i in range(9):
            if tablero[i] == 2:
                tablero[i] = otro_jugador
                valor = maximizar(tablero, jugador)
                tablero[i] = 2
                mejor_valor = min(mejor_valor, valor)
        return mejor_valor

    mejor_movimiento = -1
    mejor_valor = -math.inf
    for i in range(9):
        if tablero[i] == 2:
            tablero[i] = jugador
            valor = minimizar(tablero, jugador)
            tablero[i] = 2
            if valor > mejor_valor:
                mejor_valor = valor
                mejor_movimiento = i
    return mejor_movimiento

def mostrar_tablero(tablero):
    """
    Muestra el tablero con las representaciones de los jugadores.
    """
    for i in range(3):
        row = []
        for j in range(3):
            if tablero[i * 3 + j] == 2:
                row.append(" ")
            elif tablero[i * 3 + j] == 0:
                row.append("O")
            elif tablero[i * 3 + j] == 1:
                row.append("X")
        print("|".join(row))



def main():

    last_player = 1
    arrayJugada = []  # Inicializamos arrayJugada
    # Abrir el archivo de video
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("./WIN_20240401_23_24_55_Pro.mp4")
    #WIN_20240401_23_24_55_Pro.mp4

    while cap.isOpened():
        # Leer un fotograma del video
        ret, frame = cap.read()

        if not ret:
            break  # Si no se puede leer más fotogramas, salir del bucle

        # Procesar el fotograma actual
        squares = detect_squares(frame)
        largest_square = filter_largest_square(squares)

        if largest_square is not None:
            x, y, w, h = cv2.boundingRect(largest_square)
            roi = frame[y:y + h, x:x + w]

            # Detectar cuadrados pequeños dentro del ROI
            small_squares = detect_small_squares(roi)
            filtered_small_squares = filter_small_squares(small_squares)
            tolerancia_y = 10

            # Ordenar los cuadrados priorizando la coordenada x, pero considerando la diferencia en y
            filtered_small_squares_ordenado = sorted(filtered_small_squares, key=lambda square: (cv2.boundingRect(square)[1] // tolerancia_y, cv2.boundingRect(square)[0]))

            if len(filtered_small_squares_ordenado) ==9:
                jugadaActual = []
                for idx, small_square in enumerate(filtered_small_squares_ordenado):
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

                    if arrayJugada != jugadaActual and len(jugadaActual) ==9:
                        arrayJugada = jugadaActual
                        calcularJugada(arrayJugada, last_player) 

        # Mostrar el fotograma procesado
        cv2.imshow('Frame', frame)

        # Salir del bucle si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
