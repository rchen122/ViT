import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from models import vit
import argparse
import yaml


def testLoader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))  # CIFAR-100 stats
    ])
    test_dataset = torchvision.datasets.CIFAR100(root='dataset/', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=2)
    return test_dataset, test_loader


def test_loop(config, load_model):
    model_config = config["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = vit.ViT(**model_config)
    model.to(device)
    model.eval()

    if load_model:
        checkpoint = torch.load(load_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint from {load_model}")
    else:
        print("No model checkpoint provided. Exiting.")
        return

    loss_fn = nn.CrossEntropyLoss()
    _, test_loader = testLoader()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    print(f"Test Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--load-model', type=str, required=True, help="Path to trained model checkpoint")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    test_loop(config, args.load_model)
