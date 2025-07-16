
# 🧠 Real-Time Face Mask Detection using CNN (MobileNetV2)

This project is a real-time face mask detection system using a Convolutional Neural Network (CNN) based on MobileNetV2, trained on annotated face images to detect:
- 😷 Faces with masks
- 😐 Faces without masks
- 😕 Faces wearing masks incorrectly

---

## 🚀 Features

- ✅ Trained on annotated dataset with XML bounding boxes
- ✅ Classifier based on MobileNetV2 for lightweight deployment
- ✅ Class weight balancing to improve performance on rare classes
- ✅ Tested on both static images and real-time webcam input
- ✅ Built in Python with TensorFlow/Keras and OpenCV

---

## 🧪 Dataset Used

- [Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)  
- Contains:  
  - 3,232 images with masks  
  - 717 without masks  
  - 123 with mask worn incorrectly  
- Format: Pascal VOC XML annotations

---

## 📉 Real-World Limitations

| Challenge | Description |
|----------|-------------|
| ⚠️ **Bias in dataset** | Overrepresentation of masked faces caused overprediction of “with_mask” |
| ⚠️ **Misclassification of bearded faces** | Some bearded non-masked faces were detected as “with_mask” |
| ⚠️ **No face detection** | CNN model assumes input is already a cropped face — needs face detector for real-world use |
| ⚠️ **Not robust to occlusions or backlighting** | Faces with scarves, hands, or poor lighting affect performance |

---

## ✅ Pros and Cons

| Pros | Cons |
|------|------|
| ✅ Lightweight MobileNetV2 runs on CPU or GPU | ❌ Needs improvement on diverse real-world faces |
| ✅ Can be deployed on Colab or locally | ❌ Can't detect multiple faces in one image |
| ✅ Simple to extend with new classes | ❌ Doesn’t localize masks (only classifies face crops) |

---

## 🛠️ How to Run

### 1. 🧰 Requirements

Install dependencies:

```bash
pip install tensorflow opencv-python scikit-learn matplotlib
````

If using GUI:

```bash
pip install pillow
```

### 2. 🗂️ Dataset Setup

Place the following structure in your Google Drive or local machine:

```
/archive/
├── images/
│   └── *.png
└── annotations/
    └── *.xml
```

Make sure the annotation XML files are in **Pascal VOC format**.

### 3. ▶️ Training (Google Colab Recommended)

In `train_mask_model.py`:

* Preprocess dataset
* Resize using OpenCV
* Apply class weights
* Train MobileNetV2 model

### 4. 🧪 Inference (Static Image)

```python
from tensorflow.keras.models import load_model
model = load_model("mask_detector.model")

# Load and preprocess image
# Run prediction
# Display result
```

Or use the provided script:

```bash
python predict_image.py --image path/to/test.jpg
```

### 5. 🎥 Real-Time Webcam (Optional)

```bash
python face_mask_gui.py
```

> ⚠️ Requires `cv2.VideoCapture()` and working webcam.

---

## 💡 How to Improve

* ✅ Switch to **YOLOv8** for bounding-box + class detection
* ✅ Add face detection step (e.g., Haar Cascade, MTCNN, or RetinaFace)
* ✅ Augment training data with lighting/occlusion/pose variation
* ✅ Use Grad-CAM to visualize model attention
* ✅ Collect false positives and retrain (hard negative mining)

---

## 📸 Example Results

| Input Image                                | Prediction        |
| ------------------------------------------ | ----------------- |
| <img src="samples/test1.jpg" width="150"/> | ✅ Mask Detected   |
| <img src="samples/test2.jpg" width="150"/> | ❌ No Mask         |
| <img src="samples/test3.jpg" width="150"/> | ⚠️ Incorrect Mask |

---

## 📂 Folder Structure

```
.
├── train_mask_model.py
├── predict_image.py
├── face_mask_gui.py
├── mask_detector.model
├── archive/
│   ├── images/
│   └── annotations/
├── README.md
```

---

## 📜 License

MIT License. Attribution appreciated.

---

## 🤖 Future Plans

* Train YOLOv8 on this dataset for detection + classification
* Use AWS Lambda or Flask API for browser-based detection

```

---
