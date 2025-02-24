import h5py
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import sys

# Configuration
class Config:
    FEATURES_H5 = '/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5'
    QUESTIONS_H5 = '/Users/guoyuzhang/University/Y5/diss/vqa/code/h5_files/train_questions.h5'
    VOCAB_JSON = '/Users/guoyuzhang/University/Y5/diss/vqa/code/data/vocab.json'
    BATCH_SIZE = 32  # Increased batch size for efficiency
    EMBEDDING_DIM = 256
    LSTM_HIDDEN_DIM = 512
    IMAGE_FEATURE_DIM = 1024 * 14 * 14  # Flattened image features
    NUM_CLASSES = None  # To be determined from data
    PROGRAM_SEQ_LEN = 27  # Length of the program sequence
    PROGRAM_VOCAB_SIZE = None  # To be determined from data

# Custom Dataset (Same as in training script)
class VQADatasetInference(Dataset):
    def __init__(self, features_h5_path, questions_h5_path, indices):
        self.features_h5_path = features_h5_path
        self.questions_h5_path = questions_h5_path
        self.indices = indices
        self.features_file = h5py.File(self.features_h5_path, 'r')
        self.questions_file = h5py.File(self.questions_h5_path, 'r')
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
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

# Model Definition (Same as in training script)
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

# Function to evaluate program matching
def evaluate_program_matching(model, dataloader, device, program_idx_to_token, program_seq_len):
    model.eval()
    total_tokens = 0
    matched_tokens = 0
    total_programs = 0
    exact_matches = 0

    with torch.no_grad():
        for batch in dataloader:
            image_features, questions, answers, programs = batch
            image_features = image_features.to(device)
            questions = questions.to(device)
            programs = programs.to(device)

            # Forward pass
            _, outputs_program = model(image_features, questions)

            # Get predicted programs
            _, predicted_program = torch.max(outputs_program, 2)  # (batch, seq_len)
            predicted_program = predicted_program.cpu().numpy()
            ground_truth_program = programs.cpu().numpy()

            for pred_prog, gt_prog in zip(predicted_program, ground_truth_program):
                # Remove padding (assuming padding index is 0)
                pred_tokens = [token for token in pred_prog if token != 0]
                gt_tokens = [token for token in gt_prog if token != 0]

                # Update total tokens
                total_tokens += len(gt_tokens)

                # Compare token by token
                matches = sum(p == g for p, g in zip(pred_tokens, gt_tokens))
                matched_tokens += matches
                # print(pred_tokens, "\n" ,gt_tokens)
                # print(matches, len(pred_tokens), len(gt_tokens))
                # Update total programs
                total_programs += 1

                # Check for exact match
                if np.array_equal(pred_tokens, gt_tokens):
                    exact_matches += 1

    token_accuracy = matched_tokens / total_tokens * 100 if total_tokens > 0 else 0
    exact_match_accuracy = exact_matches / total_programs * 100 if total_programs > 0 else 0

    print(f"Token-Level Accuracy: {token_accuracy:.2f}%")
    print(f"Exact Program Match Accuracy: {exact_match_accuracy:.2f}%")

def main():
    # Get data info
    vocab_size, num_classes, program_vocab_size = get_data_info(Config.QUESTIONS_H5)
    Config.NUM_CLASSES = num_classes
    Config.PROGRAM_VOCAB_SIZE = program_vocab_size

    # Load the vocabulary JSON file
    with open(Config.VOCAB_JSON, 'r') as f:
        vocab = json.load(f)
        question_idx_to_token = {idx: token for token, idx in vocab['question_token_to_idx'].items()}
        answer_idx_to_token = {idx: token for token, idx in vocab['answer_token_to_idx'].items()}
        program_idx_to_token = {idx: token for token, idx in vocab['program_token_to_idx'].items()}

    # Device configuration
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Initialize the model
    model = VQAModel(
        vocab_size=vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        lstm_hidden_dim=Config.LSTM_HIDDEN_DIM,
        image_feature_dim=Config.IMAGE_FEATURE_DIM,
        num_classes=Config.NUM_CLASSES,
        program_vocab_size=Config.PROGRAM_VOCAB_SIZE,
        program_seq_len=Config.PROGRAM_SEQ_LEN
    ).to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load('models/best_lstm_iqap2.pth', map_location=device))
    model.eval()

    # Create a list of all indices
    with h5py.File(Config.QUESTIONS_H5, 'r') as f:
        total_samples = len(f['questions'])
    # print(len(list(range(total_samples))))
    all_indices = list(range(total_samples))

    # Create dataset and dataloader for inference
    inference_dataset = VQADatasetInference(Config.FEATURES_H5, Config.QUESTIONS_H5, all_indices)
    inference_loader = DataLoader(inference_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Evaluate program matching
    evaluate_program_matching(
        model=model,
        dataloader=inference_loader,
        device=device,
        program_idx_to_token=program_idx_to_token,
        program_seq_len=Config.PROGRAM_SEQ_LEN
    )

    # Optionally, if you still want to print some sample inferences:
    print("\nSample Inferences:")
    sample_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)
    for i, batch in enumerate(sample_loader):
        # if i >= 10:  # Print only first 10 samples
        #     break
        image_features, questions, answers, programs = batch
        image_features = image_features.to(device)
        questions = questions.to(device)
        answers = answers.to(device)
        programs = programs.to(device)

        with torch.no_grad():
            outputs_answer, outputs_program = model(image_features, questions)

        # Get predicted answer
        _, predicted_answer = torch.max(outputs_answer, 1)
        predicted_answer_idx = predicted_answer.item()
        predicted_answer_token = answer_idx_to_token.get(predicted_answer_idx, '<UNK>')

        # Get predicted program
        _, predicted_program = torch.max(outputs_program, 2)  # (batch, seq_len)
        predicted_program = predicted_program.squeeze(0).cpu().numpy()
        predicted_program_tokens = [program_idx_to_token.get(idx, '<UNK>') for idx in predicted_program if idx != 0]

        # Convert input question indices to tokens
        question_indices = questions.squeeze(0).cpu().numpy()
        question_tokens = [question_idx_to_token.get(idx, '<UNK>') for idx in question_indices if idx != 0]

        # Convert ground truth program indices to tokens
        program_indices = programs.squeeze(0).cpu().numpy()
        program_tokens = [program_idx_to_token.get(idx, '<UNK>') for idx in program_indices if idx != 0]

        # Convert ground truth answer index to token
        answer_idx = answers.item()
        answer_token = answer_idx_to_token.get(answer_idx, '<UNK>')

        # Output the data used in inference and the model's result
        print(f"\nSample {i + 1}:")
        print("Input Question Tokens:")
        print(' '.join(question_tokens))
        print("\nGround Truth Answer:")
        print(answer_token)
        print("Predicted Answer:")
        print(predicted_answer_token)
        print("Ground Truth Program Tokens:")
        print(' '.join(program_tokens))
        print("Predicted Program Tokens:")
        print(' '.join(predicted_program_tokens))

if __name__ == "__main__":
    main()
