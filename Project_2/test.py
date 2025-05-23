from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# from pytorch.utils.data import DataLoader, Dataset  
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic dataset
X, y = make_moons(n_samples=10000, noise=0.1, random_state=42)
X = X.astype(np.float32)
y = y.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# Define a custom dataset
class MoonDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# Create DataLoader
train_dataset = MoonDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = MoonDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define an Autoencoder with Latent Space
class LatentSpaceNN(nn.Module):
    def __init__(self, latent_dim=2):
        super(LatentSpaceNN, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)) # Latent space representation
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 2))  # Reconstruct original input
        
        # Classifier head (using latent representation)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid())
    
    def forward(self, x):
        # Get latent representation
        z = self.encoder(x)
        
        # Reconstruct input (for visualization/analysis)
        x_recon = self.decoder(z)
        
        # Classification output
        y_pred = self.classifier(z).squeeze()
        
        return y_pred, z, x_recon

# Initialize the model with latent dimension of 2 (for easy visualization)
latent_dim = 2
model = LatentSpaceNN(latent_dim)
criterion = nn.BCELoss()  # For classification
recon_criterion = nn.MSELoss()  # For reconstruction
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with both classification and reconstruction
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs, z, x_recon = model(batch_X)
        
        # Calculate losses
        class_loss = criterion(outputs, batch_y)
        recon_loss = recon_criterion(x_recon, batch_X)
        
        # Combined loss (you can adjust weights)
        loss = class_loss + 0.1 * recon_loss  # 0.1 weights reconstruction less
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.2e}')

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = []
    latent_representations = []
    for batch_X, _ in test_loader:
        outputs, z, _ = model(batch_X)
        y_pred.append(outputs)
        latent_representations.append(z)
    
    y_pred = torch.cat(y_pred).numpy()
    latent_representations = torch.cat(latent_representations).numpy()

# Convert predictions to binary
y_pred_binary = (y_pred > 0.5).astype(int)

# Plotting the results
plt.figure(figsize=(18, 6))

# True labels
plt.subplot(1, 3, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', alpha=0.5)
plt.title('True Labels')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Predicted labels
plt.subplot(1, 3, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_binary, cmap='coolwarm', alpha=0.5)
plt.title('Predicted Labels')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Latent space visualization
plt.subplot(1, 3, 3)
plt.scatter(latent_representations[:, 0], latent_representations[:, 1], 
            c=y_test[:len(latent_representations)], cmap='coolwarm', alpha=0.5)
plt.title('Latent Space Representation')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')

plt.tight_layout()
plt.show()
