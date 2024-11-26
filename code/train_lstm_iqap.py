import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
class Config:
    FEATURES_H5 = '/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5'
    QUESTIONS_H5 = '/Users/guoyuzhang/University/Y5/diss/code/h5_files/train_questions.h5'
    BATCH_SIZE = 64
    EMBEDDING_DIM = 256
    LSTM_HIDDEN_DIM = 512
    IMAGE_FEATURE_DIM = 1024 * 14 * 14  # Flattened image features
    NUM_CLASSES = None  # To be determined from data
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    VALIDATION_SPLIT = 0.1
    TEST_SPLIT = 0.1
    SEED = 42
    PROGRAM_SEQ_LEN = 27  # Length of the program sequence
    PROGRAM_VOCAB_SIZE = None  # To be determined from data

torch.manual_seed(Config.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Custom Dataset
class VQADataset(Dataset):
    def __init__(self, features_h5_path, questions_h5_path, indices):
        self.features_h5_path = features_h5_path
        self.questions_h5_path = questions_h5_path
        self.indices = indices
        self.features_file = None
        self.questions_file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        if self.features_file is None:
            self.features_file = h5py.File(self.features_h5_path, 'r')
        if self.questions_file is None:
            self.questions_file = h5py.File(self.questions_h5_path, 'r')
        
        # Retrieve image index
        image_idx = self.questions_file['image_idxs'][actual_idx]
        image_features = self.features_file['features'][image_idx]  # Shape: (1024, 14, 14)
        image_features = torch.tensor(image_features, dtype=torch.float32).view(-1)  # Flatten to (1024*14*14,)

        # Retrieve question
        question = self.questions_file['questions'][actual_idx]  # Shape: (46,)
        question = torch.tensor(question, dtype=torch.long)

        # Retrieve answer
        answer = self.questions_file['answers'][actual_idx]
        answer = torch.tensor(answer, dtype=torch.long)

        # Retrieve program
        program = self.questions_file['programs'][actual_idx]  # Shape: (27,)
        program = torch.tensor(program, dtype=torch.long)

        return image_features, question, answer, program

    def __del__(self):
        if self.features_file is not None:
            self.features_file.close()
        if self.questions_file is not None:
            self.questions_file.close()

# Model Definition
class VQAModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, image_feature_dim, num_classes, program_vocab_size, program_seq_len):
        super(VQAModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True)
        self.image_fc = nn.Linear(image_feature_dim, lstm_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Answer classifier
        self.classifier = nn.Linear(lstm_hidden_dim * 2, num_classes)
        
        # Program decoder
        self.program_seq_len = program_seq_len
        self.program_vocab_size = program_vocab_size
        self.program_decoder_fc = nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim)
        self.program_decoder_lstm = nn.LSTM(lstm_hidden_dim, lstm_hidden_dim, batch_first=True)
        self.program_output = nn.Linear(lstm_hidden_dim, program_vocab_size)

    def forward(self, image_features, questions, program_targets=None):
        # Encode questions
        embedded = self.embedding(questions)  # (batch, seq_len, embedding_dim)
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # h_n: (1, batch, hidden_dim)
        question_encoding = h_n.squeeze(0)  # (batch, hidden_dim)

        # Process image features
        image_encoded = self.image_fc(image_features)  # (batch, hidden_dim)
        image_encoded = self.relu(image_encoded)

        # Combine encodings
        combined = torch.cat((question_encoding, image_encoded), dim=1)  # (batch, hidden_dim * 2)
        combined = self.dropout(combined)

        # Answer prediction
        answer_output = self.classifier(combined)  # (batch, num_classes)

        # Program decoding
        # Initialize decoder hidden state
        program_decoder_input = self.program_decoder_fc(combined)  # (batch, hidden_dim)
        program_decoder_input = self.relu(program_decoder_input)
        program_decoder_input = program_decoder_input.unsqueeze(1)  # (batch, 1, hidden_dim)

        # Repeat the input for each time step
        program_decoder_input = program_decoder_input.repeat(1, self.program_seq_len, 1)  # (batch, seq_len, hidden_dim)

        # Pass through LSTM
        program_decoder_output, _ = self.program_decoder_lstm(program_decoder_input)  # (batch, seq_len, hidden_dim)

        # Generate program tokens
        program_output = self.program_output(program_decoder_output)  # (batch, seq_len, program_vocab_size)

        return answer_output, program_output

# Utility function to get the vocabulary size and number of classes
def get_data_info(questions_h5_path):
    with h5py.File(questions_h5_path, 'r') as f:
        questions = f['questions']
        answers = f['answers']
        programs = f['programs']
        vocab_size = int(np.max(questions)) + 1  # Assuming 0 is padding
        num_classes = int(np.max(answers)) + 1
        program_vocab_size = int(np.max(programs)) + 1
    return vocab_size, num_classes, program_vocab_size

# Training and Evaluation Functions
def train_epoch(model, dataloader, criterion_answer, criterion_program, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_answer = 0
    correct_program = 0
    correct_tokens = 0
    total = 0
    total_tokens = 0

    for image_features, questions, answers, programs in tqdm(dataloader, desc="Training", leave=False):
        image_features = image_features.to(device)
        questions = questions.to(device)
        answers = answers.to(device)
        programs = programs.to(device)

        optimizer.zero_grad()
        outputs_answer, outputs_program = model(image_features, questions, programs)

        # Compute answer loss
        loss_answer = criterion_answer(outputs_answer, answers)

        # Compute program loss
        outputs_program_reshaped = outputs_program.view(-1, outputs_program.size(-1))
        programs_reshaped = programs.view(-1)
        loss_program = criterion_program(outputs_program_reshaped, programs_reshaped)

        # Total loss
        loss = loss_answer + loss_program
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * image_features.size(0)

        # Compute answer accuracy
        _, predicted_answer = torch.max(outputs_answer, 1)
        correct_answer += (predicted_answer == answers).sum().item()

        # Compute program accuracy (exact match)
        _, predicted_program = torch.max(outputs_program, 2)  # (batch, seq_len)
        exact_matches = (predicted_program == programs).all(dim=1).sum().item()
        correct_program += exact_matches

        # Compute token-wise accuracy
        correct_tokens += (predicted_program == programs).sum().item()
        total_tokens += programs.numel()

        total += answers.size(0)

    epoch_loss = running_loss / total
    epoch_acc_answer = correct_answer / total
    epoch_acc_program = correct_program / total
    epoch_token_acc = correct_tokens / total_tokens

    return epoch_loss, epoch_acc_answer, epoch_acc_program, epoch_token_acc

def evaluate(model, dataloader, criterion_answer, criterion_program, device):
    model.eval()
    running_loss = 0.0
    correct_answer = 0
    correct_program = 0
    correct_tokens = 0
    total = 0
    total_tokens = 0

    with torch.no_grad():
        for image_features, questions, answers, programs in tqdm(dataloader, desc="Evaluating", leave=False):
            image_features = image_features.to(device)
            questions = questions.to(device)
            answers = answers.to(device)
            programs = programs.to(device)

            outputs_answer, outputs_program = model(image_features, questions, programs)

            # Compute answer loss
            loss_answer = criterion_answer(outputs_answer, answers)

            # Compute program loss
            outputs_program_reshaped = outputs_program.view(-1, outputs_program.size(-1))
            programs_reshaped = programs.view(-1)
            loss_program = criterion_program(outputs_program_reshaped, programs_reshaped)

            # Total loss
            loss = loss_answer + loss_program
            running_loss += loss.item() * image_features.size(0)

            # Compute answer accuracy
            _, predicted_answer = torch.max(outputs_answer, 1)
            correct_answer += (predicted_answer == answers).sum().item()

            # Compute program accuracy (exact match)
            _, predicted_program = torch.max(outputs_program, 2)  # (batch, seq_len)
            exact_matches = (predicted_program == programs).all(dim=1).sum().item()
            correct_program += exact_matches

            # Compute token-wise accuracy
            correct_tokens += (predicted_program == programs).sum().item()
            total_tokens += programs.numel()

            total += answers.size(0)

    epoch_loss = running_loss / total
    epoch_acc_answer = correct_answer / total
    epoch_acc_program = correct_program / total
    epoch_token_acc = correct_tokens / total_tokens

    return epoch_loss, epoch_acc_answer, epoch_acc_program, epoch_token_acc

# Main Training Loop
def main():
    # Get data info
    vocab_size, num_classes, program_vocab_size = get_data_info(Config.QUESTIONS_H5)
    Config.NUM_CLASSES = num_classes
    Config.PROGRAM_VOCAB_SIZE = program_vocab_size
    print(f"Vocab Size: {vocab_size}, Number of Classes: {num_classes}, Program Vocab Size: {program_vocab_size}")

    # Create dataset indices
    total_samples = 699989
    indices = list(range(total_samples))
    
    # Split indices into train, val, test
    train_val_indices, test_indices = train_test_split(
        indices, test_size=Config.TEST_SPLIT, random_state=Config.SEED)
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=Config.VALIDATION_SPLIT / (1 - Config.TEST_SPLIT), random_state=Config.SEED)

    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}, Test samples: {len(test_indices)}")

    # Create datasets
    train_dataset = VQADataset(Config.FEATURES_H5, Config.QUESTIONS_H5, train_indices)
    val_dataset = VQADataset(Config.FEATURES_H5, Config.QUESTIONS_H5, val_indices)
    test_dataset = VQADataset(Config.FEATURES_H5, Config.QUESTIONS_H5, test_indices)

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

    model = VQAModel(
        vocab_size=vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        lstm_hidden_dim=Config.LSTM_HIDDEN_DIM,
        image_feature_dim=Config.IMAGE_FEATURE_DIM,
        num_classes=Config.NUM_CLASSES,
        program_vocab_size=Config.PROGRAM_VOCAB_SIZE,
        program_seq_len=Config.PROGRAM_SEQ_LEN
    ).to(device)

    criterion_answer = nn.CrossEntropyLoss()
    criterion_program = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")

        # Training
        train_loss, train_acc_answer, train_acc_program, train_token_acc = train_epoch(
            model, train_loader, criterion_answer, criterion_program, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, "
              f"Train Acc Answer: {train_acc_answer:.4f}, "
              f"Train Acc Program: {train_acc_program:.4f}, "
              f"Train Token Acc: {train_token_acc:.4f}")

        # Validation
        val_loss, val_acc_answer, val_acc_program, val_token_acc = evaluate(
            model, val_loader, criterion_answer, criterion_program, device)
        print(f"Val Loss: {val_loss:.4f}, "
              f"Val Acc Answer: {val_acc_answer:.4f}, "
              f"Val Acc Program: {val_acc_program:.4f}, "
              f"Val Token Acc: {val_token_acc:.4f}")

        # Save the best model based on validation answer accuracy
        if val_acc_answer > best_val_acc:
            best_val_acc = val_acc_answer
            torch.save(model.state_dict(), 'best_resnet101_lstm_iqap_model.pth')
            print("Best model saved.")

    # Load the best model for testing
    model.load_state_dict(torch.load('best_resnet101_lstm_iqap_model.pth'))
    test_loss, test_acc_answer, test_acc_program, test_token_acc = evaluate(
        model, test_loader, criterion_answer, criterion_program, device)
    print(f"\nTest Loss: {test_loss:.4f}, "
          f"Test Acc Answer: {test_acc_answer:.4f}, "
          f"Test Acc Program: {test_acc_program:.4f}, "
          f"Test Token Acc: {test_token_acc:.4f}")

if __name__ == "__main__":
    main()
