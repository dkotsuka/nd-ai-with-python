import torch
from torch import nn, optim

from get_train_input_args import get_train_input_args
from get_data_loaders import get_data_loaders
from train_classifier import train_classifier
from save_checkpoint import save_checkpoint
from get_model import get_model
from load_checkpoint import load_checkpoint


def main():

    args = get_train_input_args()

    use_gpu = torch.cuda.is_available() and args.gpu
    device = torch.device('cuda' if use_gpu else 'cpu')

    trainloader, validloader, class_to_idx = get_data_loaders(args.data_dir)

    model = get_model(args.arch)
    model.class_to_idx = class_to_idx

    criterion = nn.CrossEntropyLoss()

    print('Selected architecture: {}'.format(args.arch))
    print('Selected learning rate: {}'.format(args.learning_rate))

    if (args.arch == 'vgg'):
        parameters = model.classifier.parameters()
    elif (args.arch == 'resnet'):
        parameters = model.fc.parameters()
    optimizer = optim.SGD(parameters, lr=args.learning_rate)

    epoch_counter = 0

    model, optimizer, epoch_counter = load_checkpoint(
        'checkpoint-{}.pth'.format(args.arch))

    epoch_counter += train_classifier(model, trainloader, validloader,
                                      criterion, optimizer, args.epochs, device)

    save_checkpoint(model, args.save_dir, args.arch,
                    class_to_idx, args.learning_rate, optimizer, epoch_counter)


if __name__ == '__main__':
    main()
