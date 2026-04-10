import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch import nn
import json
import math
import numpy as np


#from model import TransformerRegressionModel
#from data import generate_synthetic_data
# Hyperparameters
NUM_SAMPLES = 1000
SEQ_LEN = 64
NUM_FEATURES = 3
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2

FILE = '_dataset_3.json'

def pad_list_with_zeros(lst, target_length):
    """
    Pads the given list with zeros until it reaches the target length.

    Args:
        lst (list): The original list.
        target_length (int): Desired length after padding.

    Returns:
        list: A new list padded with zeros if needed.
    """
    # Validate inputs
    if not isinstance(lst, list):
        raise TypeError("Input must be a list.")
    if not isinstance(target_length, int) or target_length < 0:
        raise ValueError("Target length must be a non-negative integer.")

    current_length = len(lst)

    if current_length >= target_length:
        return lst[:target_length]  # Return a copy if already long enough

    # Calculate how many zeros to add
    zeros_to_add = target_length - current_length
    return lst + [0] * zeros_to_add

def _data():
    f = []
    f_mask = []
    p = []
    t = []
    # read data from json file as arrays of dictionary
    with open(FILE, 'r') as file:
    # Step 2: Parse JSON data into a Python object
        data = json.load(file)

        for item in data:
            bond = pad_list_with_zeros(list(map(int, item['bond'].split(','))), SEQ_LEN)
            bond_mask = [False if val !=0 else True for val in bond]
            a_x = pad_list_with_zeros(list(map(int, item['x'].split(','))), SEQ_LEN)
            #a_mask = [False if val !=0 else True for val in a]s
            #b=pad_list_with_zeros(item['y'].split(','),FEATURE)
            a_y = pad_list_with_zeros(list(map(int, item['y'].split(','))), SEQ_LEN)
            a_a1 = pad_list_with_zeros(list(map(int, item['a1'].split(','))), SEQ_LEN)
            a_a2 = pad_list_with_zeros(list(map(int, item['a2'].split(','))), SEQ_LEN)
            feature = list(zip(bond, a_a1, a_a2))
            f.append(feature)
            f_mask.append(bond_mask)
            position = list(zip(a_x, a_y))
            p.append(position)
            t.append(item['logp'])
        num_samples = len(p)
        #feature  = np.array(p).reshape(sample, FEATURE_DIM, 3).astype(np.float32)
        #position = np.array(p).reshape(sample, FEATURE_DIM, 2).astype(np.float32)
        #target = np.array(t).reshape(sample,1).astype(np.float32)
    #return np.array(f).astype(np.float32), np.array(p).astype(np.float32), np.array(t).astype(np.float32)
    X = np.array(f).reshape(num_samples, SEQ_LEN, 3).astype(np.float32)
    #self.y = np.array(t).astype(np.float32)
    y = np.array(t).reshape(num_samples).astype(np.float32)
    positions = np.array(p).reshape(num_samples, SEQ_LEN, 2).astype(np.float32)
    #return X, y, positions
    return torch.from_numpy(X).float(), torch.from_numpy(positions).float(),  torch.Tensor(f_mask).bool(),torch.from_numpy(y).float()

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_tensor):
        # pos_tensor: (batch, seq_len)
        freqs = torch.einsum("bi,j->bij", pos_tensor, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.x_rotary = RotaryEmbedding(dim // 2)
        self.y_rotary = RotaryEmbedding(dim // 2)

    def forward(self, positions):
        # positions: (batch, seq_len, 2)
        x_pos = positions[..., 0]
        y_pos = positions[..., 1]
        
        x_emb = self.x_rotary(x_pos)
        y_emb = self.y_rotary(y_pos)
        
        return torch.cat((x_emb, y_emb), dim=-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, q, k):
    # pos: (batch, seq_len, dim)
    # q, k: (batch, seq_len, heads, dim)
    
    pos = pos.unsqueeze(2) # (batch, seq_len, 1, dim)
    
    cos_pos = pos.cos()
    sin_pos = pos.sin()
    
    q_rot = (q * cos_pos) + (rotate_half(q) * sin_pos)
    k_rot = (k * cos_pos) + (rotate_half(k) * sin_pos)
    
    return q_rot, k_rot

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rotary_emb = RotaryEmbedding2D(self.head_dim)

    def forward(self, x, positions, key_padding_mask=None):
        batch_size, seq_len, _ = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        pos_emb = self.rotary_emb(positions)
        q, k = apply_rotary_pos_emb(pos_emb, q, k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # The key_padding_mask needs to be broadcastable to the attention matrix.
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
        else:
            attn_mask = None

        attn_output = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(attn_output)

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, positions, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2 = self.self_attn(src, positions, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerRegressionModel(nn.Module):
    def __init__(self, num_features, embed_dim, num_heads, num_layers, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Linear(num_features, embed_dim)
        
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * seq_len, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x, positions, padding_mask=None):
        x = self.embedding(x)
        for layer in self.encoder_layers:
            x = layer(x, positions, src_key_padding_mask=padding_mask)
        x = x.flatten(start_dim=1)
        x = self.regressor(x)
        return x

def train():
    # Generate data
    #X, positions, padding_mask, y = generate_synthetic_data(NUM_SAMPLES, SEQ_LEN, NUM_FEATURES)
    X, positions, padding_mask, y = _data()
    # Split data
    split_idx = int(NUM_SAMPLES * (1 - TEST_SPLIT))
    
    train_X, test_X = X[:split_idx], X[split_idx:]
    train_pos, test_pos = positions[:split_idx], positions[split_idx:]
    train_mask, test_mask = padding_mask[:split_idx], padding_mask[split_idx:]
    train_y, test_y = y[:split_idx], y[split_idx:]

    train_dataset = TensorDataset(train_X, train_pos, train_mask, train_y)
    test_dataset = TensorDataset(test_X, test_pos, test_mask, test_y)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    # Model, loss, and optimizer
    model = TransformerRegressionModel(
        num_features=NUM_FEATURES,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        seq_len=SEQ_LEN
    )
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    test_losses = []

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_pos, batch_mask, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X, batch_pos, padding_mask=batch_mask).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation on test set for each epoch
        model.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for batch_X, batch_pos, batch_mask, batch_y in test_loader:
                outputs = model(batch_X, batch_pos, padding_mask=batch_mask).squeeze()
                loss = criterion(outputs, batch_y)
                epoch_test_loss += loss.item()
        
        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    # Plotting loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()
    print("\nSaved loss plot to loss_plot.png")

    # Plotting predictions vs. true values
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch_X, batch_pos, batch_mask, batch_y in test_loader:
            outputs = model(batch_X, batch_pos, padding_mask=batch_mask)
            all_preds.extend(outputs.squeeze().tolist())
            all_true.extend(batch_y.squeeze().tolist())

    plt.figure(figsize=(8, 8))
    plt.scatter(all_true, all_preds, alpha=0.5)
    min_val = min(min(all_true), min(all_preds))
    max_val = max(max(all_true), max(all_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs. True Values on Test Set')
    plt.axis('equal')
    plt.axis('square')
    plt.savefig('predictions_plot.png')
    plt.close()
    print("Saved predictions plot to predictions_plot.png")


if __name__ == "__main__":
    train()