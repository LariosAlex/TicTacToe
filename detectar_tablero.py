import cv2
import numpy as np

def detect_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    return lines

def filter_lines(lines):
    horizontal_lines = []
    vertical_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
            
            # Horizontal line
            if abs(angle) < 10 or abs(angle) > 170:
                horizontal_lines.append(line)
            
            # Vertical line
            elif abs(angle) > 80 and abs(angle) < 100:
                vertical_lines.append(line)
    
    return horizontal_lines, vertical_lines

def find_intersections(horizontal_lines, vertical_lines):
    intersections = []
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            h_x1, h_y1, h_x2, h_y2 = h_line[0]
            v_x1, v_y1, v_x2, v_y2 = v_line[0]
            
            dx = h_x2 - h_x1
            dy = h_y2 - h_y1
            det = dx * (v_y2 - v_y1) - dy * (v_x2 - v_x1)
            
            if det != 0:
                xi = ((v_x2 - v_x1) * (h_x1 * h_y2 - h_x2 * h_y1) - dx * (v_x1 * v_y2 - v_x2 * v_y1)) / det
                yi = ((v_y2 - v_y1) * (h_x1 * h_y2 - h_x2 * h_y1) - dy * (v_x1 * v_y2 - v_x2 * v_y1)) / det
                intersections.append((int(xi), int(yi)))
    
    return intersections

def draw_lines_and_intersections(frame, horizontal_lines, vertical_lines, intersections):
    for line in horizontal_lines + vertical_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    for point in intersections:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lines = detect_lines(frame)
        horizontal_lines, vertical_lines = filter_lines(lines)
        intersections = find_intersections(horizontal_lines, vertical_lines)
        draw_lines_and_intersections(frame, horizontal_lines, vertical_lines, intersections)

        cv2.imshow("Tic Tac Toe Board Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

