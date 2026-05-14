import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.base_engine import BaseEngine
import os

class MLPModel(nn.Module):
    """MLP is a Vanilla Multi-Layer Perceptron model."""
    def __init__(self, input_dim, num_classes,
                 dropout_rate=0, hidden_dimension=50, device_id=-1):
        super(MLPModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)

        self.device_id = device_id


        hidden_sizes = [hidden_dimension, hidden_dimension]

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(in_features=hidden_sizes[0], out_features=hidden_sizes[1]),
            nn.ReLU()
        )

        # Final layer
        self.final_layer = nn.Linear(in_features=hidden_sizes[-1], out_features=num_classes)
        
        # Setup for loss calculation
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        if device_id >= 0:
            self.cuda(device=self.device_id)
            
        self.nll_loss = nn.NLLLoss()

        # Initialize weights
        self.initialize_weights()

    def forward_with_log(self, X_dense):
        hidden = self.feature_extractor(X_dense)
        logits = self.final_layer(hidden)
        y = self.log_softmax_layer(logits)
        return y

    def forward(self, X_dense):
        hidden = self.feature_extractor(X_dense)
        logits = self.final_layer(hidden)
        softmax = nn.Softmax(dim=1)
        y = softmax(logits)
        return y

    def initialize_weights(self):
        # Xavier initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.xavier_uniform_(module.bias.view(-1, 1))
    
    # TO-DO: Ensure GPU support. Not really a priority
    # def tensor_ensure_gpu(self, tensor):
    #     # Ensure tensor is on GPU if GPU is used
    #     return tensor.cuda(self.device_id) if self.device_id >= 0 else tensor



class TorchDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X (numpy.ndarray): A 2D numpy array of features.
            y (numpy.ndarray): A 1D or 2D numpy array of labels.
        """
        # Convert numpy arrays to torch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32 if y.ndim == 2 else torch.long)
        
    def __len__(self):
        # Return the size of the dataset
        return len(self.X)

    def __getitem__(self, index):
        # Support the indexing such that dataset[i] can be used to get i-th sample
        return self.X[index], self.y[index]

class MLPEngine(BaseEngine):
    
    def __init__(self, *args, **kwargs):
        super(MLPEngine, self).__init__(*args, **kwargs)
        self.model = MLPModel(*args, **kwargs)

    def predict(self, X_dense):
        self.model.eval()
        X_dense = torch.tensor(np.atleast_2d(X_dense), dtype=torch.float32)
        outputs_tensor = self.model.forward(X_dense)
        return np.array(outputs_tensor.detach())

    def train(self, X, y, X_dev=None, y_dev=None, batch_size=1000, epochs=300):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        train_dataset = TorchDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_accuracy = 0.0
        best_model_state = None

        for epoch in range(epochs):
            self.model.train()  # Ensure the model is in training mode
            total_loss = 0
            total_batches = 0

            for data, targets in train_loader:
                if self.model.device_id >= 0:
                    data, targets = data.cuda(), targets.cuda()

                optimizer.zero_grad()
                outputs = self.model.forward_with_log(data)
                loss = self.model.nll_loss(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()  # Accumulate the loss from each batch
                total_batches += 1  # Count the batch

            average_loss = total_loss / total_batches  # Calculate average loss for the epoch
            print(f"Epoch {epoch+1}, Average Loss: {average_loss:.4f}")

            if X_dev is not None:
                accuracy = self.evaluate(X_dev, y_dev)
                print(f"Dev Set Accuracy: {accuracy:.4f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_state = self.model.state_dict()  # Save the best model state

        # Revert to the best model state after training
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Model reverted to the version with highest dev set accuracy: {best_accuracy:.4f}")


    def evaluate(self, X, y):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        if self.model.device_id >= 0:
            X, y = X.cuda(), y.cuda()

        with torch.no_grad():
            outputs = self.model(X)
            predicted_labels = outputs.argmax(dim=1)
            accuracy = (predicted_labels == y).sum().item() / len(y)

        
        return accuracy


    def save(self, file_name=None):
        if file_name is None:
            file_name = "model_weights.pth"
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        torch.save(self.model.state_dict(), file_path)
        print(f"Model weights saved to {file_path}")

    def load(self, file_name):
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        self.model.load_state_dict(torch.load(file_path))
        # print(f"Model weights loaded from {file_path}")
        