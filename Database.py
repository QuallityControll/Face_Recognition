import pickle
import numpy as np


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
    with open(file_path, 'rb') as f:
        return pickle.load(f)