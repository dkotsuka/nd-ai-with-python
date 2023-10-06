import torch

from validate_classifier import validate_classifier


def train_classifier(model, pretrained, trainloader, validationloader, criterion, optimizer, epochs=5, device='cpu'):

    steps = 0
    running_loss = 0

    print_every = 10

    model = model.to(device)

    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            steps += 1

            images = images.to(device)
            features = pretrained(images)

            labels = labels.to(device)

            optimizer.zero_grad()

            output = model.forward(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                model.eval()

                with torch.no_grad():
                    test_loss, accuracy = validate_classifier(
                        model, pretrained, validationloader, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Batch: {}/{}.. ".format(steps, len(trainloader)),
                      "Training Loss: {:.3f}.. ".format(
                          running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(
                          test_loss/len(validationloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(validationloader)))

                running_loss = 0

                model.train()
