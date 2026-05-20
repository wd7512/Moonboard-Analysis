import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from moonboard_analysis.config import AutoencoderConfig
from moonboard_analysis.data.dataset import AutoencoderDataset
from moonboard_analysis.models.autoencoder import Autoencoder
from moonboard_analysis.utils.device import get_device


def train_autoencoder(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    config: AutoencoderConfig,
    device: torch.device | None = None,
) -> tuple[Autoencoder, torch.device]:
    if device is None:
        device = get_device()

    train_dataset = AutoencoderDataset(train_features)
    test_dataset = AutoencoderDataset(test_features)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    model = Autoencoder(config.input_dim, config.bottleneck_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_test_loss = float("inf")
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                loss = criterion(model(batch), batch)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss

    return model, device


def train_lstm_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for sequences, grades in train_loader:
        sequences, grades = sequences.to(device), grades.to(device)
        optimizer.zero_grad()
        loss = criterion(model(sequences), grades)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_lstm(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, grades in test_loader:
            sequences, grades = sequences.to(device), grades.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, grades)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += grades.size(0)
            correct += (predicted == grades).sum().item()
    return total_loss / len(test_loader), correct / total
