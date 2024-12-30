# to run the code: fastapi dev api.py
from typing import Annotated
from fastapi import FastAPI, File, UploadFile

# STEP 1: Import the necessary modules. 모델
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object. 추론 객체
base_options = python.BaseOptions(model_asset_path='models\\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(base_options=base_options, max_results=4)
classifier = vision.ImageClassifier.create_from_options(options)


app = FastAPI()

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

import cv2
import numpy as np
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # open file -> binary
    # http file -> text ===> binary로 변환 필요
    contents = await file.read()    # 동시 요청 가능하게 함
    nparr = np.fromstring(contents, np.uint8)

    # STEP 3: Load the input image.
    # cv_mat = cv2.imread(input_file) # imread() == 1. file open + 2. image decode
    cv_mat = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)
    # image = mp.Image.create_from_file('burger.jpg')

    # STEP 4: Classify the input image.
    classification_result = classifier.classify(image)  # forward(), inference(), get(), ...
    # print(classification_result)

    # STEP 5: Process the classification result. In this case, visualize it.
    top_category = classification_result.classifications[0].categories[0]
    result = f"{top_category.category_name} ({top_category.score:.2f})"

    return {"result": result}
