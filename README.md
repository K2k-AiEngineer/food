# 🍔 Food vs Non-Food Image Classifier 🥗

This project implements a **Food vs Non-Food Image Classifier** using **FastAPI** for creating an API that takes an image as input and predicts whether the image is of food or non-food. Multiple models are used for prediction, including CNN, ResNet, MobileNet, SVM, and Random Forest.

## 🚀 Project Overview

- **Technologies Used**:
  - **FastAPI**: For building the API.
  - **TensorFlow/Keras**: For CNN, ResNet, and MobileNet models.
  - **Scikit-learn**: For SVM and Random Forest models.
  - **OpenCV**: For image processing.
  - **PCA**: Dimensionality reduction for SVM and Random Forest models.
  
- **API Route**:
  - `/predict`: Upload an image and get predictions for food vs non-food classification.

- **Render URL** for testing:  
  - [Test the API here](https://food-non-food.onrender.com/predict)

## 🏗️ Project Structure

```bash
├── models/                   # Directory containing the saved models
│   ├── cnn_model.h5           # CNN model
│   ├── resnet_model.h5        # ResNet model
│   ├── mobilenet_model.h5     # MobileNet model
│   ├── svm_model.pkl          # SVM model (joblib format)
│   └── rf_model.pkl           # Random Forest model (joblib format)
├── main.py                    # FastAPI main app code
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

# ⚙️ How to Run Locally

### Clone the repository:
git clone https://github.com/yourusername/food-vs-non-food.git
cd food-vs-non-food

### Install dependencies:
pip install -r requirements.txt

###Run the FastAPI app
uvicorn main:app --reload


