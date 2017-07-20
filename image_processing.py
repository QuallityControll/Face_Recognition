import dlib_models
from dlib_models import load_dlib_models
import numpy as np
import skimage.io as io
from camera import take_picture
import matplotlib.pyplot as plt
from face_recog_database import update_descriptor, find_match\

load_dlib_models()
from dlib_models import models

person_database = {}

tolerance = 0.45

def camera_detect():
    pic_array = take_picture()
    detect_faces(pic_array,person_database)

def imagefile_detect(file_path):
    img_array = io.imread(file_path)
    detect_faces(file_path,person_database)

# read a picture in as a numpy-array
img_array = io.imread("pic_file_path")  # ensure this has shape (H, W, 3)
                                                  # otherwise, fix it!

def detect_faces(img_array,person_database):
    """
    Given an image, return names and descriptor vectors of found people

    :param:
        img_array:np.array
            picture to be processed

        person_database:dict
            database of names and descriptor keys

    :return:
        tuple(names,descriptors,detections)
            names:list
            descriptors:list
            detections:list
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
        names.append(find_match(person_database, descriptors[i], tolerance))

    return names, descriptors, detections
