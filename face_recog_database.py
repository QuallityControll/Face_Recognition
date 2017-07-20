import numpy as np

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
        average_descriptor = 0.5 * (d[person_name] + new_descriptor)
        d[person_name] = average_descriptor
    else:
        person_database[person_name] = new_descriptor


def find_match(d, descriptor, tolerance=0):
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
        distance = np.sqrt(np.sum((descriptor - vector)**2)) #L2 distance
        if distance < best_match_distance:
            best_match_distance = distance
            best_match_name = name

    if best_match_distance < tolerance:
        return best_match_name
    else:
        return "Person not recognized."
