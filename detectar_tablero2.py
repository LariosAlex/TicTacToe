import cv2
import numpy as np

def detect_squares(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    squares = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return squares

def filter_squares(squares):
    filtered_squares = []
    
    if squares is not None:
        for square in squares[0]:
            epsilon = 0.02 * cv2.arcLength(square, True)
            approx = cv2.approxPolyDP(square, epsilon, True)
            if len(approx) == 4:  # Cuadrado
                filtered_squares.append(approx)
    
    return filtered_squares

def draw_squares(frame, squares):
    for square in squares:
        cv2.drawContours(frame, [square], 0, (255, 0, 0), 2)

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        squares = detect_squares(frame)
        filtered_squares = filter_squares(squares)
        draw_squares(frame, filtered_squares)

        cv2.imshow("Square Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
