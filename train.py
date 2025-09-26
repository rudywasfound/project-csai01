import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer_classifier import TransformerTextClassifier
from models.axiom_model import AxiomModel
from data.dataloaders import get_ag_news_loaders, get_image_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X, y in loader:
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == "__main__":
    # Example: Run Transformer on AG_NEWS
    train_loader, test_loader, vocab = get_ag_news_loaders()
    model = TransformerTextClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        nhead=4,
        hidden_dim=256,
        num_layers=2,
        num_classes=4,
        pad_idx=vocab["<pad>"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        loss = train_epoch(model, train_loader, criterion, optimizer)
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}: Loss={loss:.4f} | Test Accuracy={acc:.4f}")

    # Example: Run AxiomModel on images (uncomment to use)
    # train_loader, test_loader = get_image_loaders()
    # model = AxiomModel(input_dim=28*28, output_dim=10).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # for epoch in range(5):
    #     loss = train_epoch(model, train_loader, criterion, optimizer)
    #     acc = evaluate(model, test_loader)
    #     print(f"[AxiomModel] Epoch {epoch+1}: Loss={loss:.4f} | Test Accuracy={acc:.4f}")
