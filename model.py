# Sound event detection using transfer learning with a pretrained model
# The classes are: 'whistle', 'cetaceans_allfreq', 'click', 'allfreq'
# The model is a pretrained mobilenet with a custom head
# The model is trained on the 4 classes above

import torch
import torchmetrics

class SubmarineAudioModel(torch.nn.Module):
    '''
    Sound event detection using transfer learning with a pretrained mobilenet model 
    There are 4 classes: 'whistle', 'cetaceans_allfreq', 'click', 'allfreq'
    The model is a pretrained mobilenet with a custom head
    '''
    def __init__(self):
        super().__init__()
        # Split the model into the feature extractor (mobilenet) and the classifier (custom head)
        self.feature_extractor = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1280, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 4)
        )

    def forward(self, x):
        # Get the features from the feature extractor
        x = self.feature_extractor.features(x)
        # Global average pooling
        x = x.mean([2, 3])
        # Get the logits from the classifier
        x = self.classifier(x)
        return x

    def predict(self, x):
        # Get the logits from the model
        logits = self.forward(x)
        # Get the class with the highest probability
        return logits.argmax(dim=1)    

    def evaluate(self, test_loader):
        # Evaluate the model with the accuracy and the F1 score macro
        self.eval()
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x, y in test_loader:
                # Get the predictions from the model
                predictions = self.predict(x)
                # Calculate the number of correct predictions
                correct += (predictions == y).sum().item()
                # Calculate the total number of predictions
                total += len(y)
                # Get the true labels
                y_true.extend(y)
                # Get the predicted labels
                y_pred.extend(predictions)
        # Calculate the accuracy
        accuracy = correct / total
        # Calculate the F1 score macro
        f1_score = torchmetrics.functional.classification.f1(y_true, y_pred, num_classes=test_loader.dataset.num_classes, average='macro')
        print(f'Accuracy: {accuracy:.4f}, F1 score macro: {f1_score:.4f}')


    def fit(self, train_loader, test_loader, epochs=10, lr=0.001):
        # setting device on GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)
        #Additional Info when using cuda
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        # Move the model to the device
        self.to(self.device)

        # Define the loss function
        criterion = torch.nn.CrossEntropyLoss()
        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # Train the model
        for epoch in range(epochs):
            # Train
            self.train()
            # Print the progress
            print(f'Epoch: {epoch+1}/{epochs})', end='\n')
            for x, y in train_loader:
                # Get the logits from the model
                logits = self.forward(x)
                # Calculate the loss
                loss = criterion(logits, y)
                print(f'Loss: {loss.item():.4f}', end='\n')
                # Zero the gradients
                optimizer.zero_grad()
                # Backpropagation
                loss.backward()
                # Update the weights
                optimizer.step()
            try:
                # Evaluate the model
                self.evaluate(test_loader)
            except:
                pass
        
        # Save the model
        torch.save(self.state_dict(), 'model.pth')