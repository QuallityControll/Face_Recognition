from .image_processing import *
from .face_recog_database import *
import skimage.io as io
from camera import take_picture
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
import os

person_database = dict()

__all__ = ["add_picture", "process_image", "process_camera", "display_names_and_boxes", "identify",
            "add", "remove", "list_people"]

def add_image(name,img_array):
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

    detections, shapes, descriptors = detect_faces(person_database,img_array)

    if len(descriptors)==0:
        print("No people found.")
    elif len(descriptors)>1:
        print("Multiple people detected. Picture can only contain one person.")
    else:
        update_descriptor(person_database, name, *descriptors)
        print("Person successfully added.")

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
        ax.text(l, b, names[i], color="white")

def identify(file_path = None, display_picture = False):
    """
    This function acts as a shortcut for the user. Given a file path,
    the image will be processed to identify any people. Given no argument,
    the camera will be used.

    :param:
        file_path : str
            String representation of path to image file.
    """
    if file_path is None:
        print("No file path found. Taking picture.")
        img_array, names, detections, shapes, descriptors = process_camera()
    else:
        img_array, names, detections, shapes, descriptors = process_image(file_path)
    
    if display_picture:
            display_names_and_boxes(img_array,names, detections)
    return names

def add(name = None, file_path = None, folder = False):
    """
    This function acts as a shortcut for the user. If a name is not entered,
    the function will prompt the user for one. Given a file path,
    the image will be processed to identify any people. Given no file path,
    the camera will be used.

    :param:
        name: [str], default: None
            name of person to be identified

        file_path: [str], default: None
            path to image file

        folder: [Boolean], default: False
            if true, adds all images in folder under one name
    """
    if not folder:
        if name is None:
            name = input("No name found. Please enter your name: ")

        if file_path is None:
            print("No file path found. Taking picture.")
            img_array = take_picture()
        else:
            img_array = io.imread("pic_file_path")

        add_image(name, img_array)

    else:
        if name is None:
            name = input("No name found. Please enter name person in folder contents: ")

        if file_path is None:
            file_path = input("No file path found. Please enter file path: ")

        for filename in os.listdir(file_path):
            if filename.endswith(".pkl"):
                img_array = io.imread(filename)
                add_image(name, img_array)


def list_people():
    """
    Function lists all people currently in database.
    """

    person_list = []
    for person in person_database:
        person_list.append(person)
    return person_list

def remove(name):
    """
    This function removes a person of a given name from the database.

    :param:
        name: [str]
            name of person to be removed
    """
    del person_database[name]

def save(file_path = "database.pkl"):
    """
    This saves the database at the specified file path. If no path is specified,
    saves to a file within the package.

    :param:
        file_path: [String]
            The file path where they want to save the file
    """

    with open(file_path, 'wb') as f:
        pickle.dump(person_database, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Person database saved to " + file_path)

def load(file_path = "database.pkl"):
    """
    This loads a file as a dictionary.

    :param:
        file_path: [String]
            This is the path of the file that you want to load.
    :return:
        pickle.load(f): [Dictionary]
            This is the dictionary stored in the file path provided
    """
    while True:
        doLoad = input("Database contents will be overwritten. Proceed? (y/n): ")
        if doLoad == 'y':
            break
        elif doLoad == 'n':
            return
        else:
            continue

    with open(file_path, 'rb') as f:
        person_database = pickle.load(f)
    print("Database loaded.")
