import torch
import json

from get_predict_input_args import get_predict_input_args
from load_checkpoint import load_checkpoint


def get_prediction(image, model, device, path_to_class_json, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()

    model.to(device)
    image = image.unsqueeze(0)
    image = image.to(device)
    output = model(image)
    ps = torch.exp(output)

    idx_to_class = {idx: class_key for class_key,
                    idx in model.class_to_idx.items()}

    values, indices = torch.topk(ps, k=topk)

    with open(path_to_class_json, 'r') as f:
        cat_to_name = json.load(f, strict=False)

    values = values.cpu().detach().numpy().squeeze()
    classes = [idx_to_class[idx] for idx in indices.squeeze().tolist()]
    class_names = [cat_to_name[class_key] for class_key in classes]

    model.train()

    return values, class_names, classes
