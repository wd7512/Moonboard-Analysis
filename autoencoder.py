"""
Moonboard Autoencoder Implementation

This script implements an autoencoder neural network that compresses 
Moonboard climbing route data from 100% input dimension to 5% hidden dimension.

The autoencoder learns to compress 164-dimensional hold position vectors into 
an 8-dimensional bottleneck (5% compression), then reconstruct back to 164 dimensions.

Training is unsupervised - the model learns to reconstruct the input data,
effectively learning a compressed representation of climbing routes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class ClimbingRouteDataset(Dataset):
    """Dataset for climbing route feature vectors."""
    
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


class Autoencoder(nn.Module):
    """
    Autoencoder that compresses climbing route data from input_dim to 
    bottleneck_dim (5% of input_dim).
    
    Architecture:
    - Encoder: input_dim -> 64 -> bottleneck_dim
    - Decoder: bottleneck_dim -> 64 -> input_dim
    """
    
    def __init__(self, input_dim, bottleneck_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder: compress to bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU()
        )
        
        # Decoder: reconstruct from bottleneck
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, input_dim),
            nn.Sigmoid()  # Output in [0, 1] range for binary data
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Get the bottleneck representation."""
        return self.encoder(x)


def load_data():
    """Load and prepare the Moonboard training data."""
    print("Loading Moonboard training data...")
    
    # Load the 164-dimension data
    data = np.load('Legacy/2016TrainingData164.npy', allow_pickle=True)
    
    # Extract features (second column) and grades (first column)
    features = np.array([row[1] for row in data])
    grades = np.array([row[0] for row in data])
    
    print(f"  Loaded {len(features)} routes")
    print(f"  Feature dimension: {features.shape[1]}")
    print(f"  Grade range: {grades.min()} to {grades.max()}")
    
    return features, grades


def train_autoencoder(features, input_dim, bottleneck_dim, epochs=100, batch_size=64):
    """Train the autoencoder model."""
    
    # Split data
    train_features, test_features = train_test_split(
        features, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {len(train_features)} samples")
    print(f"Test set: {len(test_features)} samples")
    
    # Create data loaders
    train_dataset = ClimbingRouteDataset(train_features)
    test_dataset = ClimbingRouteDataset(test_features)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = Autoencoder(input_dim, bottleneck_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    train_losses = []
    test_losses = []
    
    print(f"\nTraining Autoencoder: {input_dim} -> {bottleneck_dim} ({100*bottleneck_dim/input_dim:.1f}%)")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch)
                loss = criterion(outputs, batch)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        scheduler.step(test_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    return model, train_losses, test_losses, device


def evaluate_reconstruction(model, features, device):
    """Evaluate reconstruction quality."""
    model.eval()
    
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        reconstructed = model(features_tensor)
        
        # Calculate various metrics
        mse = nn.MSELoss()(reconstructed, features_tensor).item()
        
        # Binary reconstruction accuracy (threshold at 0.5)
        binary_original = (features_tensor > 0.5).float()
        binary_reconstructed = (reconstructed > 0.5).float()
        binary_accuracy = (binary_original == binary_reconstructed).float().mean().item()
        
        # Exact match accuracy
        exact_match = (binary_original == binary_reconstructed).all(dim=1).float().mean().item()
        
    return {
        'mse': mse,
        'binary_accuracy': binary_accuracy,
        'exact_match': exact_match
    }


def visualize_representations(model, features, device, num_samples=5):
    """Visualize original vs reconstructed samples."""
    model.eval()
    
    with torch.no_grad():
        # Get a few samples
        indices = np.random.choice(len(features), num_samples, replace=False)
        sample_features = torch.tensor(features[indices], dtype=torch.float32).to(device)
        reconstructed = model(sample_features)
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2*num_samples))
        
        for i in range(num_samples):
            # Original
            axes[i, 0].bar(range(len(sample_features[i])), sample_features[i].cpu().numpy())
            axes[i, 0].set_title(f'Original Route {i+1}')
            axes[i, 0].set_xlim(0, len(sample_features[i]))
            
            # Reconstructed
            axes[i, 1].bar(range(len(reconstructed[i])), reconstructed[i].cpu().numpy())
            axes[i, 1].set_title(f'Reconstructed Route {i+1}')
            axes[i, 1].set_xlim(0, len(reconstructed[i]))
            axes[i, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('autoencoder_reconstruction.png', dpi=150, bbox_inches='tight')
        print("\nSaved reconstruction visualization to autoencoder_reconstruction.png")
        plt.close()


def plot_training_curves(train_losses, test_losses):
    """Plot training and test loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(test_losses, label='Test Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Autoencoder Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('autoencoder_training_curves.png', dpi=150, bbox_inches='tight')
    print("Saved training curves to autoencoder_training_curves.png")
    plt.close()


def main():
    """Main function to run the autoencoder."""
    
    # Configuration
    INPUT_DIM = 164   # 164 hold positions
    BOTTLENECK_DIM = 8  # 5% of 164 = 8.2 ≈ 8
    EPOCHS = 100
    BATCH_SIZE = 64
    
    print("=" * 60)
    print("Moonboard Autoencoder - 100% to 5% Compression")
    print("=" * 60)
    print(f"Input dimension: {INPUT_DIM}")
    print(f"Bottleneck dimension: {BOTTLENECK_DIM} ({100*BOTTLENECK_DIM/INPUT_DIM:.1f}%)")
    print()
    
    # Load data
    features, grades = load_data()
    
    # Train autoencoder
    model, train_losses, test_losses, device = train_autoencoder(
        features, INPUT_DIM, BOTTLENECK_DIM, EPOCHS, BATCH_SIZE
    )
    
    # Save model
    model_path = 'Autoencoder_Moonboard.pth'
    torch.save({
        'input_dim': INPUT_DIM,
        'bottleneck_dim': BOTTLENECK_DIM,
        'model_state_dict': model.state_dict(),
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Evaluate reconstruction
    print("\n" + "=" * 60)
    print("Reconstruction Quality Metrics")
    print("=" * 60)
    
    metrics = evaluate_reconstruction(model, features, device)
    print(f"  Mean Squared Error: {metrics['mse']:.6f}")
    print(f"  Binary Accuracy: {metrics['binary_accuracy']*100:.2f}%")
    print(f"  Exact Route Match: {metrics['exact_match']*100:.2f}%")
    
    # Visualizations
    plot_training_curves(train_losses, test_losses)
    visualize_representations(model, features, device)
    
    # Show some encoded representations
    print("\n" + "=" * 60)
    print("Sample Encoded Representations (Bottleneck)")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        sample = torch.tensor(features[:5], dtype=torch.float32).to(device)
        encoded = model.encode(sample)
        print(encoded.cpu().numpy())
    
    print("\n" + "=" * 60)
    print("Autoencoder training complete!")
    print("=" * 60)
    
    return model, metrics


if __name__ == "__main__":
    main()