import dlib_models
from dlib_models import load_dlib_models
import numpy as np
import skimage.io as io
from camera import take_picture
import matplotlib.pyplot as plt
from face_recog_database import update_descriptor, find_match\

load_dlib_models()
from dlib_models import models

__all__ = ["camera_detect", "imagefile_detect, detect_faces"]

person_database = {}

tolerance = 0.45

def camera_detect():
    """
    Performs detect_faces with camera input

    :return:
        tuple(detections, shapes, descriptors)
            detections:list
            shapes:list
            descriptors:list
    """
    pic_array = take_picture()
    return detect_faces(pic_array,person_database)

def imagefile_detect(file_path):
    """
    Performs detect_faces with image input
    :param:
        file_path:str
            path to image file to be processed

    :return:
        tuple(detections, shapes, descriptors)
            detections:list
            shapes:list
            descriptors:list
    """
    img_array = io.imread(file_path)
    return detect_faces(file_path,person_database)

def detect_faces(img_array,person_database):
    """
    Given an image, return names and descriptor vectors of found people

    :param:
        img_array:np.array
            picture to be processed

        person_database:dict
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
