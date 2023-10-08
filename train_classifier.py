import torch

from validate_classifier import validate_classifier


def train_classifier(model, trainloader, validloader, criterion, optimizer, epochs=5, device='cpu', print_every=20):

    running_loss = 0

    model = model.to(device)

    print("Training started using {}...".format(device))

    for e in range(epochs):
        model.train()
        steps = 0
        for images, labels in trainloader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                model.eval()

                with torch.no_grad():
                    test_loss, accuracy = validate_classifier(
                        model, validloader, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Batch: {}/{}.. ".format(steps, len(trainloader)),
                      "Training Loss: {:.3f}.. ".format(
                          running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(
                          test_loss/len(validloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                model.train()

    return epochs
