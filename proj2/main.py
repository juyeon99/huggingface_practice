# https://github.com/deepinsight/insightface/blob/master/examples/demo_analysis.py
# Face Detection

# STEP 1: import modules
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# STEP 2: Create inference object(instance)
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

# STEP 3: Load data
# img = ins_get_image('t1')
img1 = cv2.imread('./jk.jpg')
img2 = cv2.imread('./jk2.jpg')

# STEP 4: Inference
faces1 = app.get(img1)
faces2 = app.get(img2)
assert len(faces1)==1
assert len(faces2)==1

# STEP 5: Post-processing
# STEP 5-1: Save result image
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)

# STEP 5-2: Face Recognition
# then print all-to-all face similarity
# feats = []
# for face in faces:
#     feats.append(face.normed_embedding)
face_feat1 = faces1[0].normed_embedding
face_feat2 = faces2[0].normed_embedding
face_feat1 = np.array(face_feat1, dtype=np.float32)
face_feat2 = np.array(face_feat2, dtype=np.float32)
sims = np.dot(face_feat1, face_feat2.T)
print(sims)
