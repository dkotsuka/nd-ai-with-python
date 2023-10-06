import argparse


def get_predict_input_args():
    """
    Function to get and return results from the command line defined by the user.
    If no arguments are provided, default arguments are used.
    Return:
        parse_args() - command line argument data
    """

    parser = argparse.ArgumentParser()

    # required args
    parser.add_argument('image_path', type=str,
                        help='Path to image file to predict')

    # optional args
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to load checkpoint')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='File with category names')
    parser.add_argument('--top_k', type=int,
                        default=5, help='Top K most likely classes')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference')

    return parser.parse_args()
