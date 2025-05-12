import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
	def __init__(self, img_size, patch_size, in_channel, embed_size):
		super().__init__()
		self.img_size = img_size
		self.patch_size = patch_size
		self.in_channel = in_channel
		self.embed_size = embed_size

		assert img_size % patch_size == 0
		self.num_patches = (img_size // patch_size)**2
		self.conv1 = nn.Conv2d(in_channel, self.embed_size, self.patch_size, self.patch_size)
		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
		self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches+1, embed_size))
		
		nn.init.trunc_normal_(self.pos_embedding, std=0.02)
		nn.init.trunc_normal_(self.cls_token, std=0.02)
		# nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')

	def forward(self, x): #need to convert [B, 3, 32, 32] to [B, num_patches + 1, embed_size]
		x = self.conv1(x) # [B, embed_size, 8, 8]
		x = x.flatten(2) #[B, embed_size, 64]
		x = x.transpose(1, 2) # [B, 64, embed_size]

		#add clk token
		B = x.shape[0]
		cls_tokens = self.cls_token.expand(B, -1, -1)
		x = torch.cat((cls_tokens, x), dim=1) # [B, 65, embed_size]
		x = x + self.pos_embedding
		return x
	
class MultiHeadAttention(nn.Module):
	def __init__(self, num_heads, embed_size, dropout=0.1):
		super().__init__()
		self.num_heads = num_heads
		self.embed_size = embed_size
		assert embed_size % num_heads == 0

		self.head_dim = embed_size // num_heads
		self.qkv_project = nn.Linear(embed_size, 3 * embed_size)
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(embed_size, embed_size)
		
		
	def forward(self, x):
		B, N, D = x.shape
		d_k = self.head_dim
		qkv = self.qkv_project(x) #[B, N, D*3]
		qkv = qkv.reshape(B, N, 3, self.num_heads, d_k) # [B, N, 3, H, d_k]
		qkv = qkv.permute(2, 0, 3, 1, 4) # [3, B, H, N, d_k]
		Q, K, V = qkv[0], qkv[1], qkv[2] #[B, H, N, d_k]

		#Compute Attention Score
		# Attention(Q, K, V) = Softmax(QK^T / sqrt(query_size)) * V
		scores = (Q @ K.transpose(2, 3)) / (d_k ** 0.5) # [B, H, N, N]
		scores = F.softmax(scores, dim = -1)
		scores = self.dropout(scores)
		context = scores @ V # [B, H, N, d_k] 
		context = context.transpose(1, 2) # [B, N, H, d_k]
		context = context.reshape(B, N, D)

		out = self.fc(context)
		return out
	

class TransformerEncoder(nn.Module):
	def __init__(self, embed_size, dropout=0.1):
		super().__init__()
		self.embed_size = embed_size
		self.dropout = nn.Dropout(dropout)
		self.norm1 = nn.LayerNorm(embed_size)
		self.attn = MultiHeadAttention(num_heads=8, embed_size=embed_size, dropout=dropout)
		self.norm2 = nn.LayerNorm(embed_size)
		self.MLP = nn.Sequential(
			nn.Linear(embed_size, embed_size*4), 
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(embed_size*4, embed_size),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		x = x + self.attn(self.norm1(x))
		x = x + self.MLP(self.norm2(x))
		return x
	

class ViT(nn.Module):
	def __init__(self, num_class, img_size, patch_size, in_channel, embed_size, num_heads, depth, dropout=0.1):
		super().__init__()
		self.img_size = img_size
		self.patch_size = patch_size
		self.in_channel = in_channel
		self.embed_size = embed_size
		self.num_heads = num_heads
		self.dropout = nn.Dropout(dropout)
		
		self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channel, embed_size)
		self.layers = nn.ModuleList([
			TransformerEncoder(embed_size, dropout) for _ in range(depth)
		])
		self.mlp_head = nn.Sequential(
			nn.Linear(embed_size, embed_size*4), # ?
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(embed_size*4, num_class)
		)

	def forward(self, x):
		x = self.patch_embedding(x)
		for layer in self.layers:
			x = layer(x)
		cls_token_out = x[:, 0] 
		logits = self.mlp_head(cls_token_out)
		return logits