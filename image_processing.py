__all__ = ["detect_faces"]

import dlib_models
from dlib_models import load_dlib_models
import numpy as np
import matplotlib.pyplot as plt
from .face_recog_database import *

load_dlib_models()
from dlib_models import models

<<<<<<< HEAD
tolerance = 0.5
=======
tolerance = 0.45
>>>>>>> 4259b9a2c69e7883e5c2cbda821fd0e6dfb0b76f

def detect_faces(d,img_array):
    """
    Given an image, return names and descriptor vectors of found people

    :param:
        img_array:np.array
            picture to be processed

        d:dict
            database of names and descriptor keys

    :return:
        tuple(detections, shapes, descriptors)
            detections:list
            shapes:list
            descriptors:list
    """
<<<<<<< HEAD

=======
    
>>>>>>> 4259b9a2c69e7883e5c2cbda821fd0e6dfb0b76f
    face_detect = models["face detect"]
    upscale = 2

    #returns sequence of face-detections
    detections = face_detect(img_array, upscale)
    detections = list(detections)

    shape_predictor = models["shape predict"]
    face_rec_model = models["face rec"]

    shapes = []
    descriptors = []
    names = []

    for i in range(len(detections)):
        shapes.append(shape_predictor(img_array, detections[i]))
        descriptors.append(np.array(face_rec_model.compute_face_descriptor(img_array, shapes[i])))

    return detections, shapes, descriptors
