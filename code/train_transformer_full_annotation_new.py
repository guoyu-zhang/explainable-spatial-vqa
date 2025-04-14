import json
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# ---------------------------
# Positional Encoding Module
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ---------------------------
# Multi-modal Transformer Model
# ---------------------------
class MultiModalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=512, dropout=0.1, max_text_len=50, max_img_tokens=196):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Project image features: from (batch, 1024, 14, 14) -> (batch, 196, 1024) then project to (batch, 196, d_model)
        self.image_proj = nn.Linear(1024, d_model)

        # Text embedding for source and target
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_text_len + max_img_tokens)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len=max_text_len)

        # Transformer with batch_first=True
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, dropout, batch_first=True)
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, image_features, src_text, tgt_text):
        """
        image_features: tensor of shape (batch, 1024, 14, 14)
        src_text: tensor of shape (batch, src_seq_len)
        tgt_text: tensor of shape (batch, tgt_seq_len) (target tokens, shifted right)
        """
        batch_size = image_features.size(0)
        img_feat = image_features.view(batch_size, 1024, 14*14).permute(0, 2, 1)  # (batch, 196, 1024)
        img_tokens = self.image_proj(img_feat)  # (batch, 196, d_model)

        src_emb = self.text_embedding(src_text)  # (batch, src_seq_len, d_model)
        encoder_input = torch.cat([img_tokens, src_emb], dim=1)  # (batch, 196+src_seq_len, d_model)
        encoder_input = self.pos_encoder(encoder_input)

        memory = self.transformer.encoder(encoder_input)  # (batch, 196+src_seq_len, d_model)

        tgt_emb = self.text_embedding(tgt_text)  # (batch, tgt_seq_len, d_model)
        tgt_emb = self.pos_decoder(tgt_emb)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(tgt_emb.device)
        output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        output = self.output_linear(output)  # (batch, tgt_seq_len, vocab_size)
        return output

# ---------------------------
# Dataset Class
# ---------------------------
class VQADataset(Dataset):
    def __init__(self, annotated_h5_path: str, features_h5_path: str, max_src_len=50, max_tgt_len=20, subset_fraction=1.0):
        """
        annotated_h5_path: path to annotated_questions_with_vocab.h5
        features_h5_path: path to image features H5 file.
        Each sample corresponds to one program step.
        subset_fraction: fraction of the training samples to use (e.g., 0.1 for 10%).
        """
        logging.info("Loading annotated questions from HDF5 file...")
        with h5py.File(annotated_h5_path, 'r') as hf:
            questions_json = hf["questions"][()].decode("utf-8")
        self.annotated_data = json.loads(questions_json)["questions"]
        logging.info(f"Loaded {len(self.annotated_data)} annotated questions.")

        logging.info("Opening image features HDF5 file...")
        self.features_h5 = h5py.File(features_h5_path, 'r')
        self.features = self.features_h5["features"]
        logging.info("Image features loaded.")

        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.samples = []
        for q in self.annotated_data:
            image_index = q["image_index"]
            for step in q["annotated_program"]:
                src_text = (step["function"] + " " + step["input_values"]).strip()
                tgt_text = step["output_values"].strip()
                if tgt_text:
                    self.samples.append({
                        "image_index": image_index,
                        "src_text": src_text,
                        "tgt_text": tgt_text
                    })
        original_sample_count = len(self.samples)
        if subset_fraction < 1.0:
            subset_size = int(len(self.samples) * subset_fraction)
            self.samples = self.samples[:subset_size]
            logging.info(f"Using {subset_size} samples out of {original_sample_count} ({subset_fraction*100:.1f}%).")
        else:
            logging.info(f"Using all {original_sample_count} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_index = sample["image_index"]
        img_feat = torch.tensor(self.features[image_index], dtype=torch.float)  # (1024,14,14)
        src_tokens = [int(tok) for tok in sample["src_text"].split()]
        tgt_tokens = [int(tok) for tok in sample["tgt_text"].split()]
        src_tensor = torch.tensor(src_tokens[:self.max_src_len], dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_tokens[:self.max_tgt_len], dtype=torch.long)
        return img_feat, src_tensor, tgt_tensor

def collate_fn(batch):
    imgs, src_seqs, tgt_seqs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    src_lengths = [len(s) for s in src_seqs]
    tgt_lengths = [len(t) for t in tgt_seqs]
    max_src = max(src_lengths)
    max_tgt = max(tgt_lengths)
    src_padded = torch.zeros(len(src_seqs), max_src, dtype=torch.long)
    tgt_padded = torch.zeros(len(tgt_seqs), max_tgt, dtype=torch.long)
    for i, s in enumerate(src_seqs):
        src_padded[i, :len(s)] = s
    for i, t in enumerate(tgt_seqs):
        tgt_padded[i, :len(t)] = t
    return imgs, src_padded, tgt_padded

# ---------------------------
# Training Code
# ---------------------------
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0
    batch_count = 0
    for imgs, src, tgt in dataloader:
        batch_count += 1
        imgs, src, tgt = imgs.to(device), src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]
        optimizer.zero_grad()
        output = model(imgs, src, tgt_input)  # (batch, tgt_seq_len-1, vocab_size)
        output = output.reshape(-1, model.vocab_size)
        tgt_target = tgt_target.reshape(-1)
        loss = criterion(output, tgt_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * tgt_target.size(0)
        predictions = output.argmax(dim=1)
        correct_tokens += (predictions == tgt_target).sum().item()
        total_tokens += tgt_target.size(0)
        if batch_count % 50 == 0:
            logging.info(f"Processed {batch_count} batches, current batch loss: {loss.item():.4f}")
    avg_loss = total_loss / total_tokens
    accuracy = correct_tokens / total_tokens
    return avg_loss, accuracy

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0
    with torch.no_grad():
        for imgs, src, tgt in dataloader:
            imgs, src, tgt = imgs.to(device), src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]
            output = model(imgs, src, tgt_input)
            output = output.reshape(-1, model.vocab_size)
            tgt_target = tgt_target.reshape(-1)
            loss = criterion(output, tgt_target)
            total_loss += loss.item() * tgt_target.size(0)
            predictions = output.argmax(dim=1)
            correct_tokens += (predictions == tgt_target).sum().item()
            total_tokens += tgt_target.size(0)
    avg_loss = total_loss / total_tokens
    accuracy = correct_tokens / total_tokens
    return avg_loss, accuracy

def main_training():
    logging.info("Starting training process...")
    annotated_h5_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/annotated_questions_with_vocab.h5"
    features_h5_path = "/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5"
    
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    d_model = 256
    nhead = 4  # you can adjust this
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 512
    dropout = 0.1
    max_src_len = 50
    max_tgt_len = 20

    # Set subset_fraction to a value < 1.0 to use a portion of the dataset, e.g., 0.1 for 10%
    subset_fraction = 0.1  # change as desired

    logging.info("Creating dataset and dataloader...")
    dataset = VQADataset(annotated_h5_path, features_h5_path, max_src_len, max_tgt_len, subset_fraction=subset_fraction)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    logging.info(f"Dataset size: {len(dataset)} samples.")

    with open("/Users/guoyuzhang/University/Y5/diss/vqa/code/vocab.json", 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    logging.info(f"Loaded vocabulary with {vocab_size} tokens.")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    logging.info(f"Using device: {device}")
    model = MultiModalTransformer(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                                  dim_feedforward, dropout, max_src_len, max_img_tokens=196).to(device)
    logging.info("Model initialized.")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # assuming pad token index 0

    for epoch in range(1, num_epochs+1):
        logging.info(f"Epoch {epoch} starting...")
        train_loss, train_acc = train_model(model, dataloader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_model(model, dataloader, criterion, device)
        logging.info(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, " +
                     f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")
    
    torch.save(model.state_dict(), "multimodal_transformer.pth")
    logging.info("Model saved as multimodal_transformer.pth")

if __name__ == "__main__":
    main_training()
