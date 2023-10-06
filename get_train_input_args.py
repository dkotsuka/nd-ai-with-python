import argparse


def get_train_input_args():
    """
    Function to get and return results from the command line defined by the user.
    If no arguments are provided, default arguments are used.
    Return:
        parse_args() - command line argument data
    """

    parser = argparse.ArgumentParser()

    # required args
    parser.add_argument('data_dir', type=str, help='Data directory')

    # optional args
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg13',
                        help='Model architecture')
    parser.add_argument('--learning_rate', type=float,
                        default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', nargs='+', type=int,
                        default=[512], help='Hidden layers units')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for trainning')

    return parser.parse_args()
