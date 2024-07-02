import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Load dataset
def load_dataset(data_dir):
    classes = ['Yes', 'Thankyou', 'I Love You', 'Hello', 'i want to talk']
    image_paths = []
    labels = []
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.endswith(('jpg', 'jpeg', 'png')):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_idx)
    return image_paths, labels

data_dir = "C:\\Users\\saish\\OneDrive\\Desktop\\asl recognisation\\Words"
image_paths, labels = load_dataset(data_dir)

# Split dataset into training and testing sets
train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Image preprocessing function
def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [150, 150])
    image = image / 255.0
    return image

# Function to create TensorFlow datasets
def create_dataset(image_paths, labels, batch_size=32, training=False):
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    dataset = tf.data.Dataset.zip((image_ds, label_ds))
    if training:
        dataset = dataset.shuffle(buffer_size=len(image_paths)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

batch_size = 32
train_dataset = create_dataset(train_paths, train_labels, batch_size, training=True)
test_dataset = create_dataset(test_paths, test_labels, batch_size)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(set(labels)), activation='softmax')  
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])

# Callbacks for training
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

epochs = 5  
history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset,
                    callbacks=[early_stopping, checkpoint, lr_scheduler])

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_loss, test_accuracy = model.evaluate(test_dataset)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Function to predict using the trained model and camera input
def predict_from_camera(model):
    detector = HandDetector(detectionCon=0.8)
    cap = cv2.VideoCapture(1)  # Open default camera

    while True:
        success, frame = cap.read()
        if not success:
            continue
        
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally

        hands, frame = detector.findHands(frame, flipType=False)  # Detect hands without flipping type

        if hands:
            hand = hands[0]  # Get the first hand detected
            lmList = hand["lmList"]
            bbox = hand["bbox"]

            x, y, w, h = bbox
            roi = frame[y:y+h, x:x+w]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  

            roi = cv2.resize(roi, (150, 150))
            roi = roi / 255.0
            roi = np.expand_dims(roi, axis=0)  

            prediction = model.predict(roi)
            predicted_class_idx = np.argmax(prediction, axis=-1)[0]

            classes = ['Yes', 'Thankyou', 'I Love You', 'Hello', 'i want to talk']
            predicted_class = classes[predicted_class_idx]
            cv2.putText(frame, f"Predicted: {predicted_class}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ASL Gesture Recognition", frame)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Load the best model for prediction
best_model = tf.keras.models.load_model('best_model.keras')

# Call the function to start predicting from camera input
predict_from_camera(best_model)
