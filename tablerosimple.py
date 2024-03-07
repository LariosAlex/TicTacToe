import cv2
import numpy as np

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

def main():
    # Leer la imagen
    frame = cv2.imread('square.jpg')

    # Procesar la imagen
    squares = detect_squares(frame)
    largest_square = filter_largest_square(squares)

    if largest_square is not None:
        x, y, w, h = cv2.boundingRect(largest_square)
        roi = frame[y:y + h, x:x + w]

        # Detectar cuadrados pequeños dentro del ROI
        small_squares = detect_small_squares(roi)
        filtered_small_squares = filter_small_squares(small_squares)

        # Dibujar cuadrados pequeños en rojo
        draw_squares(roi, filtered_small_squares, (0, 0, 255))

        # Dibujar el cuadrado más grande en azul
        draw_square(frame, largest_square)

        cv2.imshow("ROI", roi)

    cv2.imshow("Adjusted Sensitivity Square Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
