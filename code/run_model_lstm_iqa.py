import h5py
import torch
import torch.nn as nn
import numpy as np

# Configuration
class Config:
    FEATURES_H5 = '/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5'
    QUESTIONS_H5 = '/Users/guoyuzhang/University/Y5/diss/code/h5_files/train_questions.h5'
    EMBEDDING_DIM = 256
    LSTM_HIDDEN_DIM = 512
    IMAGE_FEATURE_DIM = 1024 * 14 * 14  # Flattened image features
    NUM_CLASSES = None  # To be determined from data
    SEED = 42

torch.manual_seed(Config.SEED)

# Custom Dataset
class VQADataset(torch.utils.data.Dataset):
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
        question = self.questions_file['questions'][actual_idx]  # Shape: (seq_len,)
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

def main():
    # Get data info
    vocab_size, num_classes = get_data_info(Config.QUESTIONS_H5)
    Config.NUM_CLASSES = num_classes
    print(f"Vocab Size: {vocab_size}, Number of Classes: {num_classes}")

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model = VQAModel(
        vocab_size=vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        lstm_hidden_dim=Config.LSTM_HIDDEN_DIM,
        image_feature_dim=Config.IMAGE_FEATURE_DIM,
        num_classes=Config.NUM_CLASSES
    ).to(device)

    # Load model weights
    model.load_state_dict(torch.load('/Users/guoyuzhang/University/Y5/diss/code/best_resnet101_lstm_answer_model.pth', map_location=device))
    model.eval()

    # Load one sample of data
    index = 0  # Index of the data sample to load
    dataset = VQADataset(Config.FEATURES_H5, Config.QUESTIONS_H5, indices=[index])
    image_features, question, answer = dataset[0]

    # Prepare data for model
    image_features = image_features.unsqueeze(0).to(device)  # Add batch dimension and move to device
    question = question.unsqueeze(0).to(device)
    answer = answer.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(image_features, question)
        predicted = torch.argmax(outputs, dim=1)

    # Output the results
    print(f"Predicted Answer ID: {predicted.item()}")
    print(f"Ground Truth Answer ID: {answer.item()}")
    print(f"Question IDs: {question.squeeze(0).cpu().numpy()}")
    print(f"Image Features Shape: {image_features.shape}")

if __name__ == "__main__":
    main()
