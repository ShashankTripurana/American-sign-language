from flask import Flask, render_template, Response
import tensorflow as tf
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("C:\\Users\\saish\\OneDrive\\Desktop\\asl recognisation\\best_model.keras")

# Class labels
classes = ['Yes', 'Thankyou',  'I Love You', 'Hello','i want to talk']

# Initialize hand detector
detector = HandDetector(detectionCon=0.8)

def preprocess_image(roi):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # Convert to RGB
    roi = cv2.resize(roi, (150, 150))
    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=0)  # Add batch dimension
    return roi

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break
        else:
            hands, frame = detector.findHands(frame, flipType=False)
            if hands:
                hand = hands[0]  # Assume single hand detection
                bbox = hand["bbox"]
                x, y, w, h = bbox
                roi = frame[y:y+h, x:x+w]
                roi = preprocess_image(roi)
                prediction = model.predict(roi)
                predicted_class_idx = np.argmax(prediction, axis=-1)[0]
                predicted_class = classes[predicted_class_idx]
                cv2.putText(frame, f"Predicted: {predicted_class}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
