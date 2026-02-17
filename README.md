# Face Mask Detector

A computer-vision project that detects whether a person is wearing a face mask. It supports single-image prediction, group-photo detection (highlighting masked faces), and a live webcam stream.

This repo includes **two app options**:
- **Streamlit app** in `app.py`
- **Flask app** in `app2.py`

## Features
- Train a CNN model for mask detection (notebook)
- Image upload inference
- Group-photo detection with face bounding boxes
- Live camera detection (Flask)

## Project Structure
```
face recognition/
├─ app.py                  # Streamlit app
├─ app2.py                 # Flask app
├─ face_mask.ipynb         # Training + experimentation notebook
├─ face_mask_detector_model.h5
├─ templates/
│  ├─ index.html
│  └─ camera.html
└─ requirements.txt
```

## Tech Stack
- Python 3
- TensorFlow / Keras (CNN model training and inference)
- OpenCV (image processing, face detection, bounding boxes)
- Streamlit (web UI for image upload)
- Flask (web UI for image upload and live camera)
- NumPy (array operations and preprocessing)
- Jupyter Notebook (experiments and training)
- HTML (Flask templates)

## Setup
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Ensure the trained model exists at `face_mask_detector_model.h5`.

## Streamlit App (app.py)
Run the Streamlit UI:
```
streamlit run app.py
```
Open the URL shown in the terminal to upload images or take a photo.

## Flask App (app2.py)
Run the Flask UI:
```
python app2.py
```
Open:
- `http://127.0.0.1:5000/` for image upload
- `http://127.0.0.1:5000/camera` for live camera

## Dataset (Kaggle)
This project uses the Face Mask Dataset from Kaggle:
```
https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
```

## Notes
- Group detection uses OpenCV Haar cascades.
- Model input size is `128x128` RGB with values scaled to `[0, 1]`.
- For best results, keep faces centered and well-lit.
