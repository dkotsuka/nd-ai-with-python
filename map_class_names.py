import json


def map_class_names(json_file, ps):
    """ Function to map class names to the output of the model.

    Args:
        json_file: path to the json file containing the class names
    """

    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)

    classes = [cat_to_name[str(idx)] for idx in ps.squeeze().tolist()]

    return classes
