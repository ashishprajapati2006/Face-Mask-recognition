import io

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
import cv2

MODEL_PATH = "face_mask_detector_model.h5"
IMAGE_SIZE = (128, 128)
CLASS_NAMES = {0: "Without Mask", 1: "With Mask"}
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def main():
    st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="centered")

    st.title("Face Mask Detector")
    st.write("Upload an image or take a photo and the model will predict whether a face mask is present.")
    st.caption("Tip: This model was trained on face crops. For best results, keep faces centered and close.")

    input_mode = st.radio("Input source", ["Upload image", "Use camera"], horizontal=True)

    uploaded_file = None
    if input_mode == "Upload image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Take a photo")

    if uploaded_file is None:
        st.info("Provide an image to get started.")
        return

    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))

    st.image(image, caption="Uploaded image", use_column_width=True)

    model = load_model()

    detect_faces = st.checkbox("Detect faces in group photos", value=True)

    if detect_faces:
        rgb_image = np.array(image.convert("RGB"))
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            st.info("No faces detected. Falling back to whole-image prediction.")
        else:
            annotated = rgb_image.copy()
            masked_count = 0

            for (x, y, w, h) in faces:
                face_rgb = rgb_image[y : y + h, x : x + w]
                face_pil = Image.fromarray(face_rgb)
                input_tensor = preprocess_image(face_pil)
                prediction = model.predict(input_tensor, verbose=0)
                predicted_index = int(np.argmax(prediction, axis=1)[0])

                if predicted_index == 1:
                    masked_count += 1
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
                    cv2.putText(
                        annotated,
                        "Mask",
                        (x, max(y - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 200, 0),
                        2,
                    )

            st.subheader("Detections")
            st.write(f"Faces found: {len(faces)}")
            st.write(f"Masked faces: {masked_count}")
            st.image(annotated, caption="Masked faces highlighted", use_column_width=True)
            return

    input_tensor = preprocess_image(image)
    prediction = model.predict(input_tensor, verbose=0)
    predicted_index = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction, axis=1)[0])

    st.subheader("Prediction")
    st.write(f"Result: {CLASS_NAMES[predicted_index]}")
    st.write(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
