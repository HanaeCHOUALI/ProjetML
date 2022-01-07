import torch
from torch import nn
from tqdm import tqdm


class NeuralNetwork(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_sizes
        self.output_size = output_size

        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

    def train_model(self, trainloader, optimizer, num_epochs):
        # We enter train mode. This is useless for the linear model
        # but is important for layers such as dropout, batchnorm, ...
        self.train()
        for i in range(num_epochs):
            loop = tqdm(trainloader)
            loop.set_description(f'Training Epoch [{i + 1}/{num_epochs}]')

            # We iterate over the mini batches of our data
            for inputs, targets in loop:
                # Erase any previously stored gradient
                optimizer.zero_grad()

                outputs = self.forward(inputs)  # Forwards stage (prediction with current weights)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, torch.max(targets, 1)[1])  # loss evaluation

                loss.backward()  # Back propagation (evaluate gradients)

                # Making gradient step on the batch (this function takes care of the gradient step for us)
                optimizer.step()

    def test_model(self, testloader):
        # Do not compute gradient, since we do not need it for validation step
        with torch.no_grad():
            # We enter evaluation mode.
            self.eval()

            total = 0  # keep track of currently used samples
            running_loss = 0.0  # accumulated loss without averaging
            accuracy = 0.0  # accumulated accuracy without averaging (number of correct predictions)

            loop = tqdm(testloader)  # This is for the progress bar
            loop.set_description('Test in progress')

            # We again iterate over the batches of test data. batch_size does not play any role here
            for inputs, targets in loop:
                # Run samples through our net
                outputs = self.forward(inputs)

                # Total number of used samples
                total += inputs.shape[0]

                # Multiply loss by the batch size to erase averaging on the batch
                criterion = nn.CrossEntropyLoss()
                running_loss += inputs.shape[0] * criterion(outputs, torch.max(targets, 1)[1]).item()

                # how many correct predictions
                accuracy += (outputs.argmax(dim=1) == torch.max(targets, 1)[1]).sum().item()

                # set nice progress message
                loop.set_postfix(test_loss=(running_loss / total), test_acc=(accuracy / total))
            return running_loss / total, accuracy / total
