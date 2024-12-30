# to run the code: fastapi dev api.py
from typing import Annotated
from fastapi import FastAPI, File, UploadFile

# STEP 1: import modules
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# STEP 2: Create inference object(instance)
face = FaceAnalysis()
face.prepare(ctx_id=0, det_size=(640,640))

app = FastAPI()

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

import cv2
import numpy as np
@app.post("/uploadfile/")
async def create_upload_file(file1: UploadFile, file2: UploadFile):
    # STEP 3: Load the input image.
    contents1 = await file1.read()
    contents2 = await file2.read()

    nparr1 = np.fromstring(contents1, np.uint8)
    nparr2 = np.fromstring(contents2, np.uint8)
    
    img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    # STEP 4: Inference
    faces1 = face.get(img1)
    faces2 = face.get(img2)
    assert len(faces1)==1
    assert len(faces2)==1

    # STEP 5: Post-processing
    # STEP 5-2: Face Recognition
    # then print all-to-all face similarity
    face_feat1 = faces1[0].normed_embedding
    face_feat2 = faces2[0].normed_embedding
    face_feat1 = np.array(face_feat1, dtype=np.float32)
    face_feat2 = np.array(face_feat2, dtype=np.float32)
    sims = np.dot(face_feat1, face_feat2.T)
    print(sims)

    if sims > 0.4:
        return {"result": "동일인입니다."}
    else:
        return {"result": "동일인이 아닙니다."}
