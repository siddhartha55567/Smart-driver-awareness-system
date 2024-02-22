import cv2
import pytesseract
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, change to 1, 2, etc. for additional cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Traffic Sign Detection
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    traffic_signs = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust this threshold according to your image
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 8:  # Assuming the sign has 8 vertices (adjust as needed)
                x, y, w, h = cv2.boundingRect(approx)
                traffic_signs.append((x, y, w, h))

    # Speed Limit Detection using Tesseract OCR
    speed_limit = None
    for x, y, w, h in traffic_signs:
        roi = gray[y:y+h, x:x+w]
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        if any(char.isdigit() for char in text):
            speed_limit = int(''.join(filter(str.isdigit, text)))
            break

    # Draw detected traffic signs and display speed limit
    for x, y, w, h in traffic_signs:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if speed_limit:
            cv2.putText(frame, f"Speed Limit: {speed_limit}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Traffic Sign Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

