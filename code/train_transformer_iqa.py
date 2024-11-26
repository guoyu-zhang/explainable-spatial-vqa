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
    TRANSFORMER_DIM = 256  # Transformer model dimension
    TRANSFORMER_HEADS = 8
    TRANSFORMER_LAYERS = 6
    TRANSFORMER_FEEDFORWARD_DIM = 512
    IMAGE_FEATURE_DIM = 1024 * 14 * 14  # Flattened image features
    NUM_CLASSES = None  # To be determined from data
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4  # Typically lower for Transformers
    VALIDATION_SPLIT = 0.1
    TEST_SPLIT = 0.1
    SEED = 42
    MAX_QUESTION_LEN = 46  # As per your data

torch.manual_seed(Config.SEED)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

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
    def __init__(self, vocab_size, embedding_dim, transformer_dim, transformer_heads, transformer_layers, transformer_ff_dim, image_feature_dim, num_classes, max_seq_len):
        super(VQAModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=max_seq_len)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=transformer_heads, dim_feedforward=transformer_ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Linear layer to project image features to transformer dimension
        self.image_fc = nn.Linear(image_feature_dim, transformer_dim)
        self.image_relu = nn.ReLU()
        self.image_dropout = nn.Dropout(0.1)
        
        # Combine image features with question encoding
        self.combine_fc = nn.Linear(transformer_dim + transformer_dim, transformer_dim)
        self.combine_relu = nn.ReLU()
        self.combine_dropout = nn.Dropout(0.1)
        
        # Classifier for answers
        self.classifier = nn.Linear(transformer_dim, num_classes)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.image_fc.weight)
        nn.init.zeros_(self.image_fc.bias)
        nn.init.xavier_uniform_(self.combine_fc.weight)
        nn.init.zeros_(self.combine_fc.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, image_features, questions):
        """
        Args:
            image_features: Tensor of shape (batch_size, image_feature_dim)
            questions: Tensor of shape (batch_size, seq_len)
        Returns:
            output: Tensor of shape (batch_size, num_classes)
        """
        batch_size = image_features.size(0)
        
        # Embed questions
        embedded = self.embedding(questions)  # (batch_size, seq_len, embedding_dim)
        embedded = self.pos_encoder(embedded)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        
        # Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(embedded)  # (seq_len, batch_size, transformer_dim)
        transformer_output = transformer_output.permute(1, 0, 2)  # (batch_size, seq_len, transformer_dim)
        
        # Aggregate question encoding (e.g., mean pooling)
        question_encoding = transformer_output.mean(dim=1)  # (batch_size, transformer_dim)
        
        # Process image features
        image_encoded = self.image_fc(image_features)  # (batch_size, transformer_dim)
        image_encoded = self.image_relu(image_encoded)
        image_encoded = self.image_dropout(image_encoded)  # (batch_size, transformer_dim)
        
        # Combine question encoding with image encoding
        combined = torch.cat((question_encoding, image_encoded), dim=1)  # (batch_size, transformer_dim * 2)
        combined = self.combine_fc(combined)  # (batch_size, transformer_dim)
        combined = self.combine_relu(combined)
        combined = self.combine_dropout(combined)  # (batch_size, transformer_dim)
        
        # Predict answer
        output = self.classifier(combined)  # (batch_size, num_classes)
        
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
        outputs = model(image_features, questions)  # (batch_size, num_classes)
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
        transformer_dim=Config.TRANSFORMER_DIM,
        transformer_heads=Config.TRANSFORMER_HEADS,
        transformer_layers=Config.TRANSFORMER_LAYERS,
        transformer_ff_dim=Config.TRANSFORMER_FEEDFORWARD_DIM,
        image_feature_dim=Config.IMAGE_FEATURE_DIM,
        num_classes=Config.NUM_CLASSES,
        max_seq_len=Config.MAX_QUESTION_LEN
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")

        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_vqa_model_transformer.pth')
            print("Best model saved.")

    # Load the best model for testing
    model.load_state_dict(torch.load('best_vqa_model_transformer.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
