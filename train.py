import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models import vit
import argparse
import yaml
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_loop(config, load_model):
	train_config = config["train"]
	model_config = config["model"]
	epochs = train_config['epochs']

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = vit.ViT(**model_config)
	model.train()
	model.to(device)

	loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
	optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
	scheduler = CosineAnnealingLR(optimizer, T_max=epochs)  


	if load_model:
		checkpoint = torch.load(load_model)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch_start = checkpoint['epoch']
		min_loss = checkpoint['loss']
	else:
		min_loss = torch.inf
		epoch_start = 0

	dst = train_config['save_dst']
	print("Starting Training Loop")
	for epoch in range(epoch_start, epochs):
		running_loss = 0.0
		for images, labels in train_loader:
			images, labels = images.to(device), labels.to(device)
			output = model(images)
			loss = loss_fn(output, labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			running_loss += loss.item()
		avg_loss = running_loss / len(train_loader)
		print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
		if avg_loss < min_loss:
			min_loss = avg_loss
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'loss': loss,
				'optimizer_state_dict': optimizer.state_dict()
			}, dst)
		if epoch % 25 == 0: # every 25 runs:
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'loss': loss,
				'optimizer_state_dict': optimizer.state_dict()
			}, f'experiments/run3/{str(epoch)}_checkpoint')


def trainLoader():
	transform = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomCrop(32, padding=4),
		transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
		transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
		transforms.ToTensor(),
		transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),  # CIFAR-100 stats
		transforms.RandomErasing(p=0.25)
	])
	train_dataset = torchvision.datasets.CIFAR100(root='dataset/', train=True, download=False, transform=transform)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
	return train_dataset, train_loader

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default='config.yaml', help='Path to config file')
	parser.add_argument('--load-model', type=str, required=False, help="Loads existing model")
	args = parser.parse_args()
	load_model = args.load_model
	with open(args.config, "r") as file:
		config = yaml.safe_load(file)
	train_dataset, train_loader = trainLoader()
	train_loop(config, load_model)
