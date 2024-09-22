import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import Transformer
from utils.data_processing import create_src_mask, create_tgt_mask
from torch.utils.data import DataLoader, TensorDataset
import sentencepiece as spm

# Hyperparameters
src_vocab_size = 150
tgt_vocab_size = 150
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
dropout = 0.1
max_seq_length = 100
batch_size = 32
num_epochs = 120
learning_rate = 0.0001

# Initialize model
model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_seq_length)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

# Load data and tokenize
src_sp = spm.SentencePieceProcessor()
tgt_sp = spm.SentencePieceProcessor()
src_sp.Load('vocab/english.model')
tgt_sp.Load('vocab/french.model')

with open('data/english_sentences.txt', 'r') as f:
    src_sentences = f.readlines()
with open('data/french_sentences.txt', 'r') as f:
    tgt_sentences = f.readlines()

src_data = [src_sp.EncodeAsIds(sent.strip()) for sent in src_sentences]
tgt_data = [tgt_sp.EncodeAsIds(sent.strip()) for sent in tgt_sentences]

# Pad sequences
max_len = max(max(len(s) for s in src_data), max(len(s) for s in tgt_data))
src_data = [s + [0] * (max_len - len(s)) for s in src_data]
tgt_data = [s + [0] * (max_len - len(s)) for s in tgt_data]

src_data = torch.LongTensor(src_data)
tgt_data = torch.LongTensor(tgt_data)

# Create DataLoader
dataset = TensorDataset(src_data, tgt_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src_mask = create_src_mask(src)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask = create_tgt_mask(tgt_input)
        
        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask)
        
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_output.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

# Save the final model
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
}, 'saved_models/final_model.pth')

print("Training finished!")