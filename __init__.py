from .image_processing import *
from .face_recog_database import *
import skimage.io as io
from camera import take_picture
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


person_database = dict()

__all__ = ["add_picture", "process_image", "process_camera", "display_names_and_boxes", "identifyMe"]

def add_picture(name,file_path):
    """
    Given a name and image of one person, logs person in database.
    If person already exists, it updates, taking the average of the
    current and given descriptor vectors.
    :param:
        name:str
            name of person in image
        file_path:str
            path to file of image of person
    """

    img_array = io.imread(file_path)
    detections, shapes, descriptors = detect_faces(person_database,img_array)

    if len(descriptors)==0:
        print("No people found.")
    elif len(descriptors)>1:
        print("Multiple people detected. With current implementation, picture can only contain one person.")
    else:
        update_descriptor(person_database, name, *descriptors)

def process_image(file_path):
    """
    Given an image, tries to predict identity of person/people.
    :param:
        file_path:str
            path to file of image
    :return:
        names:list
            list of names of people recognized
    """
    img_array = io.imread(file_path)
    detections, shapes, descriptors = detect_faces(person_database,img_array)

    names = []

    for desc in descriptors:
        name = find_match(person_database, desc)
        names.append(name)

    return pic_array, names, detections, shapes, descriptors

def process_camera():
    """
    Using a picture from current camera, tries to predict identity of person/people
    :return:
        names:list
            list
    """

    pic_array = take_picture()
    detections, shapes, descriptors = detect_faces(person_database,pic_array)

    names = []

    for desc in descriptors:
        name = find_match(person_database, desc)
        names.append(name)

    return pic_array, names, detections, shapes, descriptors

def display_names_and_boxes(img_array, names, detections):
    """
    This function takes in necessary data and displays the recognized people and boxes around their faces

    :param
        img_array: [np.array]
            A numpy array that represents a picture.
        names: [list]
            List of names of found faces
        detections: [list]
            Detection objects represting faces
    """
    fig, ax = plt.subplots()
    ax.imshow(img_array)

    for i in range(len(detections)):
        det = detections[i]
        l, r, t, b = det.left(), det.right(), det.top(), det.bottom()
        ax.add_patch(patches.Rectangle((l, t), np.abs(l - r), np.abs(t - b), fill=False))
        ax.text(l, t, names[i], color="white")

def identifyMe(file_path = None):
    """
    This function acts as a shortcut for the user. Given a file path,
    the identified people will be displayed. Given no argument,
    the camera will be used.

    :param:
        file_path : str
            String representation of path to image file.
    """
    if file_path is None:
        img_array, names, detections, shapes, descriptors = process_camera()
    else:
        img_array, names, detections, shapes, descriptors = process_image(file_path)

    display_names_and_boxes(img_array,names, detections)
