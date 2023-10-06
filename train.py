import torch
from torch import nn, optim

from get_train_input_args import get_train_input_args
from get_data_loaders import get_data_loaders
from get_pretrained_model import get_pretrained_model
from build_classifier import build_classifier
from train_classifier import train_classifier
from save_checkpoint import save_checkpoint


def main():

    args = get_train_input_args()

    use_gpu = torch.cuda.is_available() & args.gpu
    device = torch.device('cuda' if use_gpu else 'cpu')

    trainloader, validloader, testloader = get_data_loaders(args.data_dir)
    pretrained = get_pretrained_model(args.arch, device)
    classifier = build_classifier(1000, args.hidden_units, 102)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=args.learning_rate)

    train_classifier(classifier, pretrained, trainloader,
                     validloader, criterion, optimizer, args.epochs, device)

    save_checkpoint(classifier, args.save_dir, args.arch)


if __name__ == '__main__':
    main()
