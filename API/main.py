from fastapi import FastAPI , File , UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()
MODEL = tf.keras.models.load_model("../models/1")
CLASS_NAMES = ["Amoeba" , "Euglena" , "Hydra" , "Paramecium" , "Rod_bacteria" , "Spherical_bacteria" , "Spiral_bacteria" , "Yeast"]

def read_file_as_image(data) -> np.ndarray:
    image= np.array(Image.open(BytesIO(data)))
    return image

@app.get("/ping")
async def ping():
    return " hello I am alive "

@app.post("/predict")
async def predict( file: UploadFile = File(...) ):
    image = read_file_as_image(await file.read())
    image = tf.image.resize(image , [256,256])
    img_batch = np.expand_dims(image,0)

    predictions = MODEL.predict(img_batch)
    index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[index]

    confidence = np.max(predictions[0])
    
    return { 
             'class' : predicted_class , 
             'confidence' : float(confidence)
           }
    

if __name__ == "__main__":
    uvicorn.run(app , host='localhost' , port=8000)