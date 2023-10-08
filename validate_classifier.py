import torch


def validate_classifier(model, validloader, criterion, device='cpu'):
    accuracy = 0
    test_loss = 0

    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy
