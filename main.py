import os
import cv2
import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained models
cnn_model = load_model('models/cnn_model.h5')
resnet_model = load_model('models/resnet_model.h5')
mobilenet_model = load_model('models/mobilenet_model.h5')
svm_model = joblib.load('models/svm_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')

# Load PCA model
pca = joblib.load('models/pca.pkl')

# Image size for resizing
IMAGE_SIZE = (128, 128)

async def preprocess_image(image: UploadFile):
    """ Preprocess the uploaded image for prediction """
    contents = await image.read()  # Read the contents of the uploaded file
    np_img = np.frombuffer(contents, np.uint8)  # Convert to numpy array
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)  # Decode image
    img = cv2.resize(img, IMAGE_SIZE)  # Resize image
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

class PredictionResult(BaseModel):
    prediction: str

@app.post("/predict", response_model=PredictionResult)
async def predict(image: UploadFile = File(...)):
    if not image:
        return JSONResponse(status_code=400, content={"error": "No image uploaded"})

    # Preprocess the image
    img = await preprocess_image(image)

    # Predictions
    predictions = []
    
    # CNN Model Prediction
    cnn_pred = cnn_model.predict(img)
    predictions.append(np.argmax(cnn_pred))  # Append 0 (food) or 1 (non-food)

    # ResNet Model Prediction
    resnet_pred = resnet_model.predict(img)
    predictions.append(np.argmax(resnet_pred))

    # MobileNet Model Prediction
    mobilenet_pred = mobilenet_model.predict(img)
    predictions.append(np.argmax(mobilenet_pred))

    # SVM Model Prediction (requires flattened image)
    img_flat = img.reshape(1, -1)
    img_pca = pca.transform(img_flat)
    svm_pred = svm_model.predict(img_pca)
    predictions.append(svm_pred[0])

    # Random Forest Prediction
    rf_pred = rf_model.predict(img_pca)
    predictions.append(rf_pred[0])

    # Majority Voting
    food_count = predictions.count(0)  # 0 represents 'food'
    non_food_count = predictions.count(1)  # 1 represents 'non-food'

    if food_count >= 3:
        result = 'food'
    else:
        result = 'non-food'

    # Return prediction result in JSON format
    return {"prediction": result}

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
