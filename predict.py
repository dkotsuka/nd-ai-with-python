import torch

from get_predict_input_args import get_predict_input_args
from process_image import process_image
from get_prediction import get_prediction
from load_checkpoint import load_checkpoint


def main():
    """ Create a command-line tool that predicts the top K classes
        for an image, where K is an optional parameter.

        Basic usage: python predict.py /path/to/image checkpoint
    """
    args = get_predict_input_args()

    use_gpu = torch.cuda.is_available() and args.gpu
    device = torch.device('cuda' if use_gpu else 'cpu')

    image = process_image(args.image_path)

    model, optimizer, epoch_counter = load_checkpoint(
        args.checkpoint_path)

    values, class_names, classes = get_prediction(
        image, model, device, args.category_names, topk=args.top_k)

    print(class_names)


if __name__ == '__main__':
    main()
