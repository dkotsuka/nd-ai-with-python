import torch


def validate_classifier(model, pretrained, testloader, criterion, device='cpu'):
    accuracy = 0
    test_loss = 0
    model = model.to(device)

    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)

        features = pretrained(images)

        output = model.forward(features)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy
