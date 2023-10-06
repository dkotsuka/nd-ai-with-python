import torch

from get_predict_input_args import get_predict_input_args
from process_image import process_image
from get_prediction import get_prediction
from map_class_names import map_class_names


def main():
    """ Create a command-line tool that predicts the top K classes
        for an image, where K is an optional parameter.

        Basic usage: python predict.py /path/to/image checkpoint
    """
    args = get_predict_input_args()

    print(args)

    use_gpu = torch.cuda.is_available() & args.gpu
    device = torch.device('cuda' if use_gpu else 'cpu')

    image = process_image(args.image_path)

    ps = get_prediction(image, device)

    topk_values, topk_indices = torch.topk(ps, k=args.top_k)
    class_names = map_class_names(args.category_names, topk_indices)

    print(class_names)


if __name__ == '__main__':
    main()
