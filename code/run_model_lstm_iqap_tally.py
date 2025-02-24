import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from tqdm import tqdm

# Configuration
class Config:
    LAPTOP_OR_CLUSTER = 'L'  # Change this depending on running on cluster or PC
    PATH = '/exports/eddie/scratch/s1808795/vqa/code/' if LAPTOP_OR_CLUSTER == 'CLUSTER' else '/Users/guoyuzhang/University/Y5/diss/vqa/code/'
    FEATURES_H5 = PATH + 'data/train_features.h5' if LAPTOP_OR_CLUSTER == 'CLUSTER' else '/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/val_features.h5'
    QUESTIONS_H5 = PATH + 'h5_files/val_questions.h5'
    MODELS_DIR = PATH + 'models'
    MODEL_NAME = PATH + 'models/best_lstm_iqap2.pth'
    BATCH_SIZE = 64
    EMBEDDING_DIM = 512
    LSTM_HIDDEN_DIM = 512
    IMAGE_FEATURE_DIM = 1024 * 14 * 14  # Flattened image features
    NUM_CLASSES = None  # To be loaded from data
    PROGRAM_SEQ_LEN = 27  # Length of the program sequence
    PROGRAM_VOCAB_SIZE = None  # To be loaded from data
    SOS_TOKEN = 1  # Replace with actual SOS token index if different
    EOS_TOKEN = 2  # Replace with actual EOS token index if different

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Custom Dataset for Inference
class VQADatasetTest(Dataset):
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

# Model Definition (Must match the Training Model)
class VQAModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, image_feature_dim, num_classes, program_vocab_size, program_seq_len, sos_token, eos_token):
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
        self.program_decoder_lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True)
        self.program_output = nn.Linear(lstm_hidden_dim, program_vocab_size)
        
        # Special tokens
        self.sos_token = sos_token
        self.eos_token = eos_token

    def forward(self, image_features, questions, program_targets=None, teacher_forcing_ratio=0.0):
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
        batch_size = image_features.size(0)
        device = image_features.device

        # Initialize decoder hidden state
        decoder_hidden = self.program_decoder_fc(combined)  # (batch, hidden_dim)
        decoder_hidden = self.relu(decoder_hidden)
        decoder_hidden = decoder_hidden.unsqueeze(0)  # (1, batch, hidden_dim)
        decoder_cell = torch.zeros_like(decoder_hidden).to(device)  # Initialize cell state to zeros

        # Initialize input token as SOS token
        input_token = torch.full((batch_size,), self.sos_token, dtype=torch.long, device=device)  # (batch,)

        # Initialize tensors to store outputs
        program_outputs = torch.zeros(batch_size, self.program_seq_len, self.program_vocab_size, device=device)

        for t in range(self.program_seq_len):
            embedded_token = self.embedding(input_token).unsqueeze(1)  # (batch, 1, embedding_dim)
            decoder_output, (decoder_hidden, decoder_cell) = self.program_decoder_lstm(embedded_token, (decoder_hidden, decoder_cell))
            logits = self.program_output(decoder_output.squeeze(1))  # (batch, program_vocab_size)
            program_outputs[:, t, :] = logits

            # Decide whether to use teacher forcing
            if program_targets is not None and np.random.random() < teacher_forcing_ratio:
                input_token = program_targets[:, t]  # Teacher forcing
            else:
                input_token = logits.argmax(1)  # Use model's own prediction

        return answer_output, program_outputs

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

# Inference Function with Tallying
def run_inference():
    """
    Runs inference on the test set and tallies the results into four categories:
    1. Both Answer and Program Correct
    2. Answer Correct but Program Incorrect
    3. Answer Incorrect but Program Correct
    4. Both Answer and Program Incorrect
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data info
    vocab_size, num_classes, program_vocab_size = get_data_info(Config.QUESTIONS_H5)
    Config.NUM_CLASSES = num_classes
    Config.PROGRAM_VOCAB_SIZE = program_vocab_size

    print(f"Vocab Size: {vocab_size}, Number of Classes: {num_classes}, Program Vocab Size: {program_vocab_size}")

    # Initialize the model
    model = VQAModel(
        vocab_size=vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        lstm_hidden_dim=Config.LSTM_HIDDEN_DIM,
        image_feature_dim=Config.IMAGE_FEATURE_DIM,
        num_classes=Config.NUM_CLASSES,
        program_vocab_size=Config.PROGRAM_VOCAB_SIZE,
        program_seq_len=Config.PROGRAM_SEQ_LEN,
        sos_token=Config.SOS_TOKEN,
        eos_token=Config.EOS_TOKEN
    ).to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load(Config.MODEL_NAME, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # Create dataset indices for testing
    # Assuming the test set was defined during training. Adjust if different.
    # For simplicity, we'll process all samples in the questions file.
    with h5py.File(Config.QUESTIONS_H5, 'r') as f_questions:
        total_samples = f_questions['questions'].shape[0]
        print(f"Total samples in test set: {total_samples}")
        test_indices = list(range(total_samples))

    # Create test dataset and dataloader
    test_dataset = VQADatasetTest(Config.FEATURES_H5, Config.QUESTIONS_H5, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize tally counters
    tally_both_correct = 0
    tally_answer_correct_program_incorrect = 0
    tally_answer_incorrect_program_correct = 0
    tally_both_incorrect = 0

    # Disable gradient computation for inference
    with torch.no_grad():
        for image_features, questions, answers, programs in tqdm(test_loader, desc="Inference"):
            image_features = image_features.to(device)  # (batch, 1024*14*14)
            questions = questions.to(device)  # (batch, question_seq_len)
            answers = answers.to(device)  # (batch,)
            programs = programs.to(device)  # (batch, program_seq_len)

            # Run the model
            answer_output, program_outputs = model(image_features, questions, program_targets=None, teacher_forcing_ratio=0.0)

            # Get predicted answers
            _, predicted_answers = torch.max(answer_output, dim=1)  # (batch,)

            # Get predicted programs
            predicted_programs = torch.argmax(program_outputs, dim=2)  # (batch, program_seq_len)

            # Compare answers
            answer_correct = (predicted_answers == answers)  # (batch,)

            # Compare programs (exact match per sample)
            program_correct = torch.all(predicted_programs == programs, dim=1)  # (batch,)

            # Update tallies
            both_correct = answer_correct & program_correct
            answer_correct_only = answer_correct & (~program_correct)
            program_correct_only = (~answer_correct) & program_correct
            both_incorrect = (~answer_correct) & (~program_correct)

            tally_both_correct += both_correct.sum().item()
            tally_answer_correct_program_incorrect += answer_correct_only.sum().item()
            tally_answer_incorrect_program_correct += program_correct_only.sum().item()
            tally_both_incorrect += both_incorrect.sum().item()

    # After processing all samples, print the tallies
    print("\n=== Inference Tally Results ===")
    print(f"Total Samples Processed: {total_samples}")
    print(f"1. Both Answer and Program Correct: {tally_both_correct}")
    print(f"2. Answer Correct but Program Incorrect: {tally_answer_correct_program_incorrect}")
    print(f"3. Answer Incorrect but Program Correct: {tally_answer_incorrect_program_correct}")
    print(f"4. Both Answer and Program Incorrect: {tally_both_incorrect}")
    print("================================\n")

    # (Optional) Calculate percentages
    print("=== Inference Tally Percentages ===")
    print(f"1. Both Correct: {tally_both_correct / total_samples * 100:.2f}%")
    print(f"2. Answer Correct, Program Incorrect: {tally_answer_correct_program_incorrect / total_samples * 100:.2f}%")
    print(f"3. Answer Incorrect, Program Correct: {tally_answer_incorrect_program_correct / total_samples * 100:.2f}%")
    print(f"4. Both Incorrect: {tally_both_incorrect / total_samples * 100:.2f}%")
    print("=====================================\n")

# Utility function to convert question indices to string (if vocabulary is available)
def question_to_string(question_indices, vocab=None):
    # Placeholder function: Replace with actual vocabulary mapping if available
    # Example:
    # return ' '.join([vocab[idx] for idx in question_indices if idx != 0])
    return ' '.join(map(str, question_indices))

# Utility function to convert answer index to string (if label mapping is available)
def answer_to_string(answer_index, label_map=None):
    # Placeholder function: Replace with actual label mapping if available
    # Example:
    # return label_map[answer_index]
    return str(answer_index)

# Utility function to convert program indices to string (if program vocabulary is available)
def program_to_string(program_indices, program_vocab=None):
    # Placeholder function: Replace with actual program vocabulary mapping if available
    # Example:
    # return ' '.join([program_vocab[idx] for idx in program_indices if idx != 0])
    return ' '.join(map(str, program_indices))

if __name__ == "__main__":
    run_inference()
