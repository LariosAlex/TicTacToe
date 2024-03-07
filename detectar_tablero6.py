import cv2
import numpy as np
import os
import time

def detect_squares(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    squares = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return squares


def filter_largest_square(squares):
    largest_square = None
    max_area = 0
   
    if squares is not None:
        for square in squares[0]:
            epsilon = 0.01 * cv2.arcLength(square, True)  # Ajusta este valor según tu sensibilidad
            approx = cv2.approxPolyDP(square, epsilon, True)
           
            if len(approx) == 4:  # Cuadrado
                area = cv2.contourArea(square)
                if area > max_area  and area > 50000: # Ajustar area al cuadrado
                    max_area = area
                    largest_square = approx
   
    return largest_square


def draw_square(frame, square):
    if square is not None:
        cv2.drawContours(frame, [square], 0, (255, 0, 0), 2)


def save_image(frame, count):
    # Crea un directorio llamado 'images' si no existe
    if not os.path.exists('images'):
        os.makedirs('images')
    
    # Guarda la imagen en el directorio 'images'
    filename = f'images/square_{count}.png'
    cv2.imwrite(filename, frame)
    print(f'Imagen guardada como {filename}')

def main():
    cap = cv2.VideoCapture(0)

    save_interval = 10  # Guardar cada 10 segundos
    last_save_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        squares = detect_squares(frame)
        largest_square = filter_largest_square(squares)
        draw_square(frame, largest_square)

        if largest_square is not None:
            # Obtén las coordenadas del cuadrado más grande
            x, y, w, h = cv2.boundingRect(largest_square)
            
            # Extrae la región de interés (ROI) del interior del cuadrado
            roi = frame[y:y + h, x:x + w]

            # Muestra la ROI en una ventana separada
            cv2.imshow("ROI", roi)

            # Guarda la imagen cada 10 segundos
            current_time = time.time()
            if current_time - last_save_time >= save_interval:
                timestamp = int(current_time)
                filename = f"square_detection_{timestamp}.png"
                cv2.imwrite(filename, frame)
                last_save_time = current_time
                print(f"Imagen guardada como {filename}")

        cv2.imshow("Adjusted Sensitivity Square Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()