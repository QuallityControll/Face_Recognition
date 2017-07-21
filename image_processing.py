__all__ = ["camera_detect", "imagefile_detect", "detect_faces"]

import dlib_models
from dlib_models import load_dlib_models
import numpy as np
import skimage.io as io
from camera import take_picture
import matplotlib.pyplot as plt
from .face_recog_database import *

load_dlib_models()
from dlib_models import models

tolerance = 0.45

def camera_detect(d):
    """
    Performs detect_faces with camera input

    :param:
        d:dict
            keys:str
                names of people
            values:np.array
                vector descriptor

    :return:
        tuple(detections, shapes, descriptors)
            detections:list
            shapes:list
            descriptors:list
    """
    pic_array = take_picture()
    return detect_faces(d,pic_array)

def imagefile_detect(d,file_path):
    """
    Performs detect_faces with image input
    :param:
        d:dict
            keys:str
                names of people
            values:np.array
                vector descriptor

        file_path:str
            path to image file to be processed

    :return:
        tuple(detections, shapes, descriptors)
            detections:list
            shapes:list
            descriptors:list
    """
    img_array = io.imread(file_path)
    return detect_faces(d,img_array)

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

    img_array = take_picture()

    face_detect = models["face detect"]
    upscale = 1

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
