from .image_processing import *
from .face_recog_database import *

__all__ = "Face_Regognition"

class Face_Recognition:
    def __init__():
        self.person_database = dict()

    def add_picture(self,name,file_path):
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

        detections, shapes, descriptors = imagefile_detect(file_path)
        if len(descriptors)==0:
            print("No people found.")
        elif len(descriptors)>1:
            print("Multiple people detected. With current implementation, picture can only contain one person.")
        else:
            update_descriptor(self.person_database, name, *descriptors)

    def process_image(self,file_path):
        """
        Given an image, tries to predict identity of person/people.
        :param:
            file_path:str
                path to file of image
        :return:
            names:list
                list of names of people recognized
        """

        detections, shapes, descriptors = imagefile_detect(file_path)

        names = []

        for desc in descriptors:
            name = find_match(self.person_database, desc)
            names.append(name)

        return names

    def process_camera(self):
        """
        Using a picture from current camera, tries to predict identity of person/people
        :return:
            names:list
                list
        """

        detections, shapes, descriptors = camera_detect()

        names = []

        for desc in descriptors:
            name = find_match(self.person_database, desc)
            names.append(name)

        return names
