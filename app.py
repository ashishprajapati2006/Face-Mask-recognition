import base64
import io
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, render_template, request
from PIL import Image
from tensorflow import keras

MODEL_PATH = Path(__file__).parent / "face_mask_detector_model.h5"
IMAGE_SIZE = (128, 128)
CLASS_NAMES = {0: "Without Mask", 1: "With Mask"}
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(FACE_CASCADE_PATH)

app = Flask(__name__)
model = keras.models.load_model(str(MODEL_PATH))


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def detect_masked_faces(image: Image.Image):
    rgb_image = np.array(image.convert("RGB"))
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None, 0, 0

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

    return annotated, len(faces), masked_count


def annotate_frame(frame_bgr: np.ndarray) -> np.ndarray:
    rgb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face_rgb = rgb_image[y : y + h, x : x + w]
        face_pil = Image.fromarray(face_rgb)
        input_tensor = preprocess_image(face_pil)
        prediction = model.predict(input_tensor, verbose=0)
        predicted_index = int(np.argmax(prediction, axis=1)[0])

        if predicted_index == 1:
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(
                frame_bgr,
                "Mask",
                (x, max(y - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),
                2,
            )

    return frame_bgr


def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            annotated = annotate_frame(frame)
            ok, buffer = cv2.imencode(".jpg", annotated)
            if not ok:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    faces_found = None
    masked_count = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("image")
        detect_faces = request.form.get("detect_faces") == "on"

        if file and file.filename:
            image = Image.open(io.BytesIO(file.read()))

            if detect_faces:
                annotated, faces_found, masked_count = detect_masked_faces(image)

                if annotated is not None:
                    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    _, buffer = cv2.imencode(".jpg", annotated_bgr)
                    image_url = "data:image/jpeg;base64," + base64.b64encode(
                        buffer.tobytes()
                    ).decode("ascii")
                    result = "Group detection complete"
                else:
                    result = "No faces detected. Falling back to whole-image prediction."

            if image_url is None:
                input_tensor = preprocess_image(image)
                prediction = model.predict(input_tensor, verbose=0)
                predicted_index = int(np.argmax(prediction, axis=1)[0])
                confidence = float(np.max(prediction, axis=1)[0])
                result = CLASS_NAMES[predicted_index]

                image_bytes = io.BytesIO()
                image.convert("RGB").save(image_bytes, format="JPEG")
                image_url = "data:image/jpeg;base64," + base64.b64encode(
                    image_bytes.getvalue()
                ).decode("ascii")

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_url=image_url,
        faces_found=faces_found,
        masked_count=masked_count,
    )


@app.route("/camera")
def camera():
    return render_template("camera.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)
