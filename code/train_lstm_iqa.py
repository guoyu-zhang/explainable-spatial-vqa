import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
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

torch.manual_seed(Config.SEED)

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
        
        # Lazy loading of H5 files
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

        return image_features, question, answer

    def __del__(self):
        if self.features_file is not None:
            self.features_file.close()
        if self.questions_file is not None:
            self.questions_file.close()

# Model Definition
class VQAModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, image_feature_dim, num_classes):
        super(VQAModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True)
        self.image_fc = nn.Linear(image_feature_dim, lstm_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(lstm_hidden_dim * 2, num_classes)

    def forward(self, image_features, questions):
        # Encode questions
        # questions is (batch, seq_len)
        embedded = self.embedding(questions)  # (batch, seq_len, embedding_dim)
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # h_n: (1, batch, hidden_dim)
        question_encoding = h_n.squeeze(0)  # (batch, hidden_dim)

        # Process image features
        image_encoded = self.image_fc(image_features)  # (batch, hidden_dim)
        image_encoded = self.relu(image_encoded)

        # Combine encodings
        combined = torch.cat((question_encoding, image_encoded), dim=1)  # (batch, hidden_dim * 2)
        combined = self.dropout(combined)
        output = self.classifier(combined)  # (batch, num_classes)
        return output

# Utility function to get the vocabulary size and number of classes
def get_data_info(questions_h5_path):
    with h5py.File(questions_h5_path, 'r') as f:
        questions = f['questions']
        answers = f['answers']
        vocab_size = int(np.max(questions)) + 1  # Assuming 0 is padding
        num_classes = int(np.max(answers)) + 1
    return vocab_size, num_classes

# Training and Evaluation Functions
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for image_features, questions, answers in tqdm(dataloader, desc="Training", leave=False):
        image_features = image_features.to(device)
        questions = questions.to(device)
        answers = answers.to(device)

        optimizer.zero_grad()
        outputs = model(image_features, questions)
        loss = criterion(outputs, answers)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * image_features.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == answers).sum().item()
        total += answers.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for image_features, questions, answers in tqdm(dataloader, desc="Evaluating", leave=False):
            image_features = image_features.to(device)
            questions = questions.to(device)
            answers = answers.to(device)

            outputs = model(image_features, questions)
            loss = criterion(outputs, answers)

            running_loss += loss.item() * image_features.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == answers).sum().item()
            total += answers.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Main Training Loop
def main():
    # Get data info
    vocab_size, num_classes = get_data_info(Config.QUESTIONS_H5)
    Config.NUM_CLASSES = num_classes
    print(f"Vocab Size: {vocab_size}, Number of Classes: {num_classes}")

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

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model = VQAModel(
        vocab_size=vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        lstm_hidden_dim=Config.LSTM_HIDDEN_DIM,
        image_feature_dim=Config.IMAGE_FEATURE_DIM,
        num_classes=Config.NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(Config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_vqa_model.pth')
            print("Best model saved.")

    # Load the best model for testing
    model.load_state_dict(torch.load('best_vqa_model.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
