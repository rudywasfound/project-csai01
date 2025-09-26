import torch
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset, random_split
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AG_NEWS Dataset
def get_ag_news_loaders(batch_size=64):
    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    train_iter = AG_NEWS(split="train")
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    def encode(text):
        return torch.tensor(vocab(tokenizer(text)), dtype=torch.long)

    def collate_batch(batch):
        label_list, text_list = [], []
        for label, text in batch:
            label_list.append(label - 1)  # labels are 1-indexed
            text_list.append(encode(text))
        padded = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
        return padded.to(device), torch.tensor(label_list, dtype=torch.long).to(device)

    train_iter, test_iter = AG_NEWS()
    train_loader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(list(test_iter), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, test_loader, vocab


# MNIST + FashionMNIST + CIFAR Combined Dataset
def get_image_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    fashion = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    cifar = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)

    combined = ConcatDataset([mnist, fashion, cifar])

    train_size = int(0.8 * len(combined))
    test_size = len(combined) - train_size
    train_ds, test_ds = random_split(combined, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
