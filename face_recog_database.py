import numpy as np
import pickle
<<<<<<< HEAD

__all__ = ["update_descriptor", "find_match"]


import numpy as np
=======
>>>>>>> 4259b9a2c69e7883e5c2cbda821fd0e6dfb0b76f

__all__ = ["update_descriptor", "find_match"]

def update_descriptor(d, person_name, new_descriptor):
    """
    Updates a vector descriptor of a given person by averaging input descriptor with current descriptor
    If given person does not exist in database, adds new descriptor.
    :param:
        d:dict
            keys:str
                names of people
            values:np.array
                vector descriptor
        person_name:str
            name of person
        new_descriptor:np.array, shape = (128,)
            vector description of person's face
    """
    if person_name in d:
        temp = d[person_name][0]*d[person_name][1] + new_descriptor
        numEntered = d[person_name][0] + 1
        average_descriptor = temp / numEntered
        d[person_name] = (numEntered,average_descriptor)
    else:
        d[person_name] = (1,new_descriptor)


def find_match(d, descriptor, tolerance=0.4):
    """
    :param:
        d:dict
            keys:str
                names of people
            values:np.array
                vector descriptor
        descriptor:np.array, shape = (128,)
            vector description of person's face
        tolerance:
    """
    best_match_name = "N/A"
    best_match_distance = 1e+300

    for name, vector in d.items():
        distance = np.sqrt(np.sum((descriptor - vector[1])**2)) #L2 distance
        if distance < best_match_distance:
            best_match_distance = distance
            best_match_name = name

    if best_match_distance < tolerance:
        return best_match_name
    else:
        return "Not recognized"

def save_data_as_file(file_path, data):
    """
    This saves the dictionary at the specified file path.

    :param:
        file_path: [String]
            The file path where they want to save the file
        data: [Dictionary]
            The dictionary with the names and descriptor vectors.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_file_as_dict(file_path):
    """
    This loads a file as a dictionary.

    :param:
        file_path: [String]
            This is the path of the file that you want to load.
    :return:
        pickle.load(f): [Dictionary]
            This is the dictionary stored in the file path provided
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)
