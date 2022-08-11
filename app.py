from tensorflow.keras.models import load_model
import numpy as np
from skimage import transform
from fastapi import FastAPI, File
from tensorflow.keras.utils import img_to_array
from PIL import Image
import tensorflow as tf
from io import BytesIO
import json
import uvicorn

app = FastAPI()


def get_model():
    cnn = load_model('cnn_model_val_accu_89.h5')
    return cnn


loaded_model = get_model()


@app.get("/")
async def root():
    return {"message": "Helmet Detection ML API"}


@app.post("/files/")
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}


@app.post("/classify/")
async def classify(file: bytes = File(...)):
    image = Image.open(BytesIO(file))
    img = img_to_array(image)
    np_image = transform.resize(img, (256, 256, 3))
    image4 = tf.expand_dims(np_image, 0)
    result__ = loaded_model.predict(image4).tolist()
    preds = np.round(result__[0], 4)
    to_List = [["Car", str(np.round(preds[1], 5))], ["PillionWithoutHelmet", str(np.round(preds[2], 5))],
               ["WithHelmet", str(np.round(preds[3], 5))], ["WithoutHelmet", str(np.round(preds[4], 5))],
               ["BothWithoutHelmet", str(np.round(preds[0], 5))]]
    return to_List


if __name__ == '__main__':
    uvicorn.run(app, port=8000, debug=True)

# host="0.0.0.0",
