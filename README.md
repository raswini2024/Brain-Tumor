# Brain Tumor Classification using MobileNetV2

This project uses deep learning and transfer learning techniques to classify brain MRI images into four categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**. The model is built using **TensorFlow** and includes a **Gradio** interface for real-time image classification.

---

##  Overview

- **Objective**: Automatically classify brain tumors from MRI images.
- **Model**: Transfer learning using **MobileNetV2**.
- **Interface**: Web-based prediction using **Gradio**.
- **Dataset**: MRI brain tumor images organized into labeled folders.

---

##  Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Scikit-learn
- Matplotlib & Seaborn
- Gradio

---

##  Dataset Structure

```

/Training/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

````

Each folder should contain MRI images of the respective class.

---

##  How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
````

### 2. Install dependencies

```bash
pip install tensorflow numpy opencv-python scikit-learn matplotlib seaborn gradio
```

>  Or run it directly in **Google Colab** with Google Drive access.

### 3. Set dataset path

Update the `DATASET_PATH` in the code:

```python
DATASET_PATH = '/content/drive/MyDrive/Brain Tumor Segmentation/Training'
```

---

## Model Summary

* Base: `MobileNetV2` (pre-trained on ImageNet)
* Added Layers:

  * `GlobalAveragePooling2D`
  * `Dense(128, activation='relu')`
  * `Dropout(0.5)`
  * `Dense(4, activation='softmax')`

---

## Training

* Image Size: `128x128`
* Batch Size: `16`
* Epochs: `25`
* Loss: `categorical_crossentropy`
* Optimizer: `Adam`
* Uses class weights to handle imbalance

---

##  Evaluation

* Accuracy and loss plots
* Classification report
* Confusion matrix visualization

---

##  Gradio Interface

Launch an interactive UI to classify brain tumor images:

```python
interface.launch()
```

Users can upload an MRI image and get predictions with confidence scores.

---

##  Output Example

```
Prediction:
- Glioma: 91%
- No Tumor: 5%
- Pituitary: 3%
- Meningioma: 1%
```

---

##  Future Work

* Save and load trained model
* Add real-time data augmentation
* Deploy as a web app using Flask or Django

