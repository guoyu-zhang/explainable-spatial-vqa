import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import argparse
import json

# Configuration
class Config:
    QUESTIONS_H5 = '/Users/guoyuzhang/University/Y5/diss/vqa/code/h5_files/train_questions.h5'
    VOCAB_PATH = '/Users/guoyuzhang/University/Y5/diss/vqa/code/data/vocab.json'  # Path to vocab.json
    BATCH_SIZE = 64
    EMBEDDING_DIM = 256
    LSTM_HIDDEN_DIM = 512
    PROGRAM_SEQ_LEN = 27  # Length of the program sequence
    PROGRAM_VOCAB_SIZE = None  # To be determined from data
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    VALIDATION_SPLIT = 0.1
    TEST_SPLIT = 0.1
    SEED = 42
    PATIENCE = 10

torch.manual_seed(Config.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Custom Dataset
class VQAProgramDataset(Dataset):
    def __init__(self, questions_h5_path, indices):
        self.questions_h5_path = questions_h5_path
        self.indices = indices
        self.questions_file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        if self.questions_file is None:
            self.questions_file = h5py.File(self.questions_h5_path, 'r')
        
        # Retrieve question
        question = self.questions_file['questions'][actual_idx]  # Shape: (46,)
        question = torch.tensor(question, dtype=torch.long)

        # Retrieve program
        program = self.questions_file['programs'][actual_idx]  # Shape: (27,)
        program = torch.tensor(program, dtype=torch.long)

        return question, program

    def __del__(self):
        if self.questions_file is not None:
            self.questions_file.close()

class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, program_vocab_size, program_seq_len, program_start_token_idx):
        super(Seq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True)
        
        self.decoder = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, program_vocab_size)
        self.program_seq_len = program_seq_len
        self.program_vocab_size = program_vocab_size
        self.program_start_token_idx = program_start_token_idx
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, questions):
        """
        Forward pass without teacher forcing.
        Generates programs step-by-step based on its own predictions.
        
        Returns:
            generated_programs: (batch, program_seq_len)
            logits_programs: (batch, program_seq_len, program_vocab_size)
        """
        # Encode questions
        embedded = self.embedding(questions)  # (batch, seq_len, embedding_dim)
        encoder_out, (hidden, cell) = self.encoder(embedded)  # hidden: (1, batch, hidden_dim)
        
        batch_size = questions.size(0)
        generated_program = torch.zeros(batch_size, self.program_seq_len, dtype=torch.long).to(questions.device)
        logits_program = torch.zeros(batch_size, self.program_seq_len, self.program_vocab_size).to(questions.device)
        input_token = torch.tensor([self.program_start_token_idx] * batch_size, dtype=torch.long).unsqueeze(1).to(questions.device)  # (batch, 1)

        hidden_dec = hidden
        cell_dec = cell

        for t in range(self.program_seq_len):
            embedded_token = self.embedding(input_token)  # (batch, 1, embedding_dim)
            decoder_out, (hidden_dec, cell_dec) = self.decoder(embedded_token, (hidden_dec, cell_dec))  # (batch, 1, hidden_dim)
            output = self.fc(decoder_out)  # (batch, 1, program_vocab_size)
            logits = output.squeeze(1)  # (batch, program_vocab_size)
            logits_program[:, t, :] = logits  # Store logits for loss computation
            _, predicted = torch.max(output, dim=2)  # (batch, 1)
            generated_program[:, t] = predicted.squeeze(1)
            input_token = predicted  # Next input is current prediction

        return generated_program, logits_program  # Return both for loss computation


# Utility function to get the vocabulary size and program vocab size
def get_data_info(questions_h5_path):
    with h5py.File(questions_h5_path, 'r') as f:
        questions = f['questions']
        programs = f['programs']
        vocab_size = int(np.max(questions)) + 1  # Assuming 0 is padding
        program_vocab_size = int(np.max(programs)) + 1
    return vocab_size, program_vocab_size

# Function to load vocabulary mappings
def load_vocab(vocab_path):
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    question_token_to_idx = vocab.get("question_token_to_idx", {})
    program_token_to_idx = vocab.get("program_token_to_idx", {})
    # Create reverse mappings
    question_idx2word = {int(idx): word for word, idx in question_token_to_idx.items()}
    program_idx2word = {int(idx): word for word, idx in program_token_to_idx.items()}
    return question_token_to_idx, question_idx2word, program_token_to_idx, program_idx2word

# Function to decode program indices to tokens
def decode_program(program_indices, program_idx2word):
    tokens = [program_idx2word.get(idx, f"<UNK:{idx}>") for idx in program_indices]
    # Optionally, stop at <END>
    if "<END>" in tokens:
        end_index = tokens.index("<END>") + 1
        tokens = tokens[:end_index]
    return ' '.join(tokens)

# Function to decode question indices to words
def decode_question(question_indices, question_idx2word):
    tokens = [question_idx2word.get(idx, f"<UNK:{idx}>") for idx in question_indices]
    # Optionally, stop at <END>
    if "<END>" in tokens:
        end_index = tokens.index("<END>") + 1
        tokens = tokens[:end_index]
    return ' '.join(tokens)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_program = 0
    correct_tokens = 0
    total = 0
    total_tokens = 0

    for questions, programs in tqdm(dataloader, desc="Training", leave=False):
        questions = questions.to(device)
        programs = programs.to(device)

        optimizer.zero_grad()
        predicted_programs, logits_programs = model(questions)  # (batch, program_seq_len), (batch, program_seq_len, vocab_size)

        # Reshape logits and targets for loss computation
        logits_reshaped = logits_programs.view(-1, logits_programs.size(-1))  # (batch * program_seq_len, vocab_size)
        programs_reshaped = programs.view(-1)  # (batch * program_seq_len)
        loss = criterion(logits_reshaped, programs_reshaped)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * questions.size(0)

        # Compute token-wise accuracy
        correct_tokens += (predicted_programs == programs).sum().item()
        total_tokens += programs.numel()

        # Compute exact match accuracy for programs
        exact_matches = (predicted_programs == programs).all(dim=1).sum().item()
        correct_program += exact_matches
        total += questions.size(0)

    epoch_loss = running_loss / total
    epoch_acc_program = correct_program / total
    epoch_token_acc = correct_tokens / total_tokens

    return epoch_loss, epoch_acc_program, epoch_token_acc

def evaluate(model, dataloader, device, question_idx2word, program_idx2word, max_print=None):
    model.eval()
    correct_program = 0
    correct_tokens = 0
    total = 0
    total_tokens = 0
    printed = 0

    with torch.no_grad():
        for questions, programs in tqdm(dataloader, desc="Evaluating", leave=False):
            questions = questions.to(device)
            programs = programs.to(device)

            # Inference: Generate predicted programs without teacher forcing
            predicted_programs, logits_programs = model(questions)  # (batch, program_seq_len), (batch, program_seq_len, vocab_size)

            # Compute token-wise accuracy
            correct_tokens += (predicted_programs == programs).sum().item()
            total_tokens += programs.numel()

            # Compute exact match accuracy for programs
            exact_matches = (predicted_programs == programs).all(dim=1).sum().item()
            correct_program += exact_matches
            total += questions.size(0)

            # Decode and print examples if within max_print
            if max_print is not None and printed >= max_print:
                continue

            batch_size = questions.size(0)
            for i in range(batch_size):
                if max_print is not None and printed >= max_print:
                    break

                # Decode question
                question_tokens = questions[i].cpu().numpy()
                question_text = decode_question(question_tokens, question_idx2word)

                # Decode ground truth program
                true_program_tokens = programs[i].cpu().numpy()
                true_program_text = decode_program(true_program_tokens, program_idx2word)

                # Decode predicted program
                pred_program_tokens = predicted_programs[i].cpu().numpy()
                pred_program_text = decode_program(pred_program_tokens, program_idx2word)

                # Print the details
                print(f"\nExample {printed + 1}:")
                print(f"Question: {question_text}")
                print(f"Ground Truth Program: {true_program_text}")
                print(f"Predicted Program: {pred_program_text}")

                printed += 1

    epoch_acc_program = correct_program / total if total > 0 else 0
    epoch_token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0

    return epoch_acc_program, epoch_token_acc

def main():
    os.makedirs('models', exist_ok=True)
    # Load vocabulary mappings
    question_token_to_idx, question_idx2word, program_token_to_idx, program_idx2word = load_vocab(Config.VOCAB_PATH)

    # Get data info
    vocab_size, program_vocab_size = get_data_info(Config.QUESTIONS_H5)
    Config.PROGRAM_VOCAB_SIZE = program_vocab_size
    print(f"Vocab Size: {vocab_size}, Program Vocab Size: {program_vocab_size}")

    # Create dataset indices
    with h5py.File(Config.QUESTIONS_H5, 'r') as f:
        total_samples = f['questions'].shape[0]
    
    indices = list(range(total_samples))
    
    # Split indices into train, val, test
    train_val_indices, test_indices = train_test_split(
        indices, test_size=Config.TEST_SPLIT, random_state=Config.SEED)
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=Config.VALIDATION_SPLIT / (1 - Config.TEST_SPLIT), random_state=Config.SEED)

    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}, Test samples: {len(test_indices)}")

    # Create datasets
    train_dataset = VQAProgramDataset(Config.QUESTIONS_H5, train_indices)
    val_dataset = VQAProgramDataset(Config.QUESTIONS_H5, val_indices)
    test_dataset = VQAProgramDataset(Config.QUESTIONS_H5, test_indices)

    # Determine the number of CPU cores for DataLoader
    import multiprocessing
    num_workers = min(4, multiprocessing.cpu_count()) 

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)

    # Initialize model, loss, optimizer
    # Device configuration
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    model = Seq2SeqModel(
        vocab_size=vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        lstm_hidden_dim=Config.LSTM_HIDDEN_DIM,
        program_vocab_size=Config.PROGRAM_VOCAB_SIZE,
        program_seq_len=Config.PROGRAM_SEQ_LEN,
        program_start_token_idx=program_token_to_idx.get("<START>", 1)  # Ensure <START> token is correctly set
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")

        # Training
        train_loss, train_acc_program, train_token_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, "
              f"Train Acc Program: {train_acc_program:.4f}, "
              f"Train Token Acc: {train_token_acc:.4f}")

        # Validation
        val_acc_program, val_token_acc = evaluate(
            model, val_loader, device, question_idx2word, program_idx2word, max_print=10)  # Adjust max_print as needed
        print(f"Val Acc Program: {val_acc_program:.4f}, "
              f"Val Token Acc: {val_token_acc:.4f}")

        # Save the best model based on validation program accuracy
        if val_acc_program > best_val_acc:
            best_val_acc = val_acc_program
            torch.save(model.state_dict(), 'models/best_seq2seq_program.pth')
            print("Best model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= Config.PATIENCE:
                print("Early stopping triggered. Stopping training.")
                break

    # Load the best model for testing
    model.load_state_dict(torch.load('models/best_seq2seq_program.pth'))
    test_acc_program, test_token_acc = evaluate(
        model, test_loader, device, question_idx2word, program_idx2word, max_print=10)  # Adjust max_print as needed
    print(f"\nTest Acc Program: {test_acc_program:.4f}, "
          f"Test Token Acc: {test_token_acc:.4f}")
   
if __name__ == "__main__":
    main()
