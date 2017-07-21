from dlib_models import load_dlib_models
import numpy as np
from camera import take_picture
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from face_recog_database import update_descriptor, find_match

load_dlib_models()
from dlib_models import models


def detect_faces_and_boxes(person_database, tolerance=0.4):
    """
    This function detects the faces and displays the image with the boxes around the faces looking at the camera
    while also putting their names if they are in the database and if not, it says not recognized

    :param
        person_database: Dictionary
            This is the dictionary of names versus the descriptor vectors
        tolerance:
            A level of tolerance of the difference between the people.
    """

    img_array = take_picture()
    fig, ax = plt.subplots()
    ax.imshow(img_array)

    face_detect = models["face detect"]
    upscale = 1

    # returns sequence of face-detections
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

    for i in range(len(detections)):
        det = detections[i]
        l, r, t, b = det.left(), det.right(), det.top(), det.bottom()
        ax.add_patch(patches.Rectangle((l, t), np.abs(l - r), np.abs(t - b), fill=False))
        ax.text(l, t, names[i], color="white")
