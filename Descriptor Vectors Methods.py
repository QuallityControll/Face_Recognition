from dlib_models import load_dlib_models
import numpy as np
from camera import take_picture
import matplotlib.pyplot as plt

load_dlib_models()

from dlib_models import models

desc = []

for i in range(2):
    img_array = take_picture()
    fig, ax = plt.subplots()
    ax.imshow(img_array)

    face_detect = models["face detect"]

    # Number of times to upscale image before detecting faces.
    # When would you want to increase this number?
    upscale = 1

    detections = face_detect(img_array, upscale)  # returns sequence of face-detections
    detections = list(detections)

    det = detections[0] # first detected face in image

    # bounding box dimensions for detection
    l, r, t, b = det.left(), det.right(), det.top(), det.bottom()

    shape_predictor = models["shape predict"]
    shape = shape_predictor(img_array, det)

    face_rec_model = models["face rec"]

    # check that shape is (128,)
    descriptor = np.array(face_rec_model.compute_face_descriptor(img_array, shape))
    desc.append(descriptor)

np.sqrt(np.sum((desc[0] - desc[1])**2))
