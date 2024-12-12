# import torch
# import torch.nn as nn
# import numpy as np
# import json
# import argparse
# import os

# # Configuration
# class Config:
#     VOCAB_PATH = '/Users/guoyuzhang/University/Y5/diss/vqa/code/data/vocab.json'  # Path to vocab.json
#     MODEL_PATH = 'models/best_seq2seq_program.pth'  # Path to the trained model
#     EMBEDDING_DIM = 256
#     LSTM_HIDDEN_DIM = 512
#     PROGRAM_SEQ_LEN = 27  # Length of the program sequence
#     QUESTION_SEQ_LEN = 46  # Length of the question sequence (as per training)
#     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')
#     SEED = 42

# torch.manual_seed(Config.SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# # Model Definition (Must match the training model)
# class Seq2SeqModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, program_vocab_size, program_seq_len, program_start_token_idx):
#         super(Seq2SeqModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.encoder = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True)
        
#         self.decoder = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True)
#         self.fc = nn.Linear(lstm_hidden_dim, program_vocab_size)
#         self.program_seq_len = program_seq_len
#         self.program_vocab_size = program_vocab_size
#         self.program_start_token_idx = program_start_token_idx
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, questions, program_targets=None):
#         # Encode questions
#         embedded = self.embedding(questions)  # (batch, seq_len, embedding_dim)
#         encoder_out, (hidden, cell) = self.encoder(embedded)  # hidden: (1, batch, hidden_dim)
        
#         if program_targets is not None:
#             # Teacher forcing: Use actual program tokens as input
#             embedded_program = self.embedding(program_targets)  # (batch, prog_seq_len, embedding_dim)
#             decoder_out, _ = self.decoder(embedded_program, (hidden, cell))  # (batch, prog_seq_len, hidden_dim)
#             program_output = self.fc(decoder_out)  # (batch, prog_seq_len, program_vocab_size)
#             return program_output
#         else:
#             # Inference mode: Generate program tokens step-by-step
#             batch_size = questions.size(0)
#             generated_program = torch.zeros(batch_size, self.program_seq_len, dtype=torch.long).to(questions.device)
#             input_token = torch.tensor([self.program_start_token_idx] * batch_size, dtype=torch.long).unsqueeze(1).to(questions.device)  # (batch, 1)
    
#             hidden_dec = hidden
#             cell_dec = cell
    
#             for t in range(self.program_seq_len):
#                 embedded_token = self.embedding(input_token)  # (batch, 1, embedding_dim)
#                 decoder_out, (hidden_dec, cell_dec) = self.decoder(embedded_token, (hidden_dec, cell_dec))  # (batch, 1, hidden_dim)
#                 output = self.fc(decoder_out)  # (batch, 1, program_vocab_size)
#                 _, predicted = torch.max(output, dim=2)  # (batch, 1)
#                 generated_program[:, t] = predicted.squeeze(1)
#                 input_token = predicted  # Next input is current prediction
    
#             return generated_program

# # Function to load vocabulary
# def load_vocab(vocab_path):
#     if not os.path.exists(vocab_path):
#         raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
#     with open(vocab_path, 'r') as f:
#         vocab = json.load(f)
#     question_token_to_idx = vocab.get("question_token_to_idx", {})
#     program_token_to_idx = vocab.get("program_token_to_idx", {})
#     # Create reverse mappings
#     question_idx2word = {int(idx): word for word, idx in question_token_to_idx.items()}
#     program_idx2word = {int(idx): word for word, idx in program_token_to_idx.items()}
#     return question_token_to_idx, question_idx2word, program_token_to_idx, program_idx2word

# # Function to preprocess the question
# def preprocess_question(question, question_token_to_idx, seq_len):
#     # Simple tokenization: split by spaces and lowercase
#     tokens = question.lower().strip().split()
#     # Convert tokens to indices, use <UNK> token if not found
#     unk_idx = question_token_to_idx.get("<UNK>", 3)  # Default to 3 if <UNK> not found
#     token_indices = [question_token_to_idx.get(token, unk_idx) for token in tokens]
#     # Truncate or pad the sequence
#     if len(token_indices) < seq_len:
#         token_indices += [question_token_to_idx["<NULL>"]] * (seq_len - len(token_indices))
#     else:
#         token_indices = token_indices[:seq_len]
#     return token_indices

# # Function to decode program indices to tokens
# def decode_program(program_indices, program_idx2word):
#     tokens = [program_idx2word.get(idx, f"<UNK:{idx}>") for idx in program_indices]
#     # Optionally, stop at <END>
#     if "<END>" in tokens:
#         end_index = tokens.index("<END>") + 1
#         tokens = tokens[:end_index]
#     return ' '.join(tokens)

# # Main Inference Function
# def main():
#     parser = argparse.ArgumentParser(description='Inference Script for Seq2Seq Program Model')
#     parser.add_argument('--question', type=str, required=False, help='Input question for which to predict the program')
#     parser.add_argument('--vocab_path', type=str, default=Config.VOCAB_PATH, help='Path to vocabulary JSON file')
#     parser.add_argument('--model_path', type=str, default=Config.MODEL_PATH, help='Path to the trained model file')
#     args = parser.parse_args()

#     # Load vocabulary
#     try:
#         (question_token_to_idx,
#          question_idx2word,
#          program_token_to_idx,
#          program_idx2word) = load_vocab(args.vocab_path)
#         print("Vocabulary loaded successfully.")
#     except Exception as e:
#         print(f"Error loading vocabulary: {e}")
#         return

#     # Determine vocab_size and program_vocab_size
#     max_question_idx = max(question_token_to_idx.values())
#     max_program_idx = max(program_token_to_idx.values())
#     vocab_size = max(max_question_idx, max_program_idx) + 1
#     program_vocab_size = len(program_token_to_idx)
#     print(f"Vocab Size: {vocab_size}, Program Vocab Size: {program_vocab_size}")

#     # Get program start token index
#     program_start_token_idx = program_token_to_idx.get("<START>", 1)  # Default to 1 if not found

#     # Initialize the model
#     model = Seq2SeqModel(
#         vocab_size=vocab_size,
#         embedding_dim=Config.EMBEDDING_DIM,
#         lstm_hidden_dim=Config.LSTM_HIDDEN_DIM,
#         program_vocab_size=program_vocab_size,
#         program_seq_len=Config.PROGRAM_SEQ_LEN,
#         program_start_token_idx=program_start_token_idx
#     ).to(Config.DEVICE)

#     # Load model weights
#     if not os.path.exists(args.model_path):
#         print(f"Model file not found at {args.model_path}")
#         return
#     try:
#         model.load_state_dict(torch.load(args.model_path, map_location=Config.DEVICE))
#         model.eval()
#         print("Model loaded successfully.")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return

#     # Check if a question is provided for single inference
#     if args.question:
#         user_question = args.question.strip()
#         if not user_question:
#             print("Empty question provided. Please provide a valid question.")
#             return

#         # Preprocess the input question
#         question_tokens = preprocess_question(user_question, question_token_to_idx, Config.QUESTION_SEQ_LEN)
#         print(question_tokens)
#         question_tensor = torch.tensor([question_tokens], dtype=torch.long).to(Config.DEVICE)  # (1, seq_len)

#         # Perform inference
#         with torch.no_grad():
#             predicted_program = model(question_tensor)  # (1, program_seq_len)
#             predicted_program = predicted_program.cpu().numpy()[0]  # (program_seq_len,)

#         print(predicted_program)
#         # Decode the predicted program
#         predicted_program_tokens = decode_program(predicted_program, program_idx2word)
        
#         # Print the results
#         print("\n--- Inference Result ---")
#         print(f"Question: {user_question}")
#         print(f"Predicted Program: {predicted_program_tokens}\n")
#     else:
#         # Interactive mode
#         print("\n--- Seq2Seq Program Model Inference ---")
#         print("Type 'exit' to quit.\n")
#         while True:
#             user_question = input("Enter a question: ")
#             if user_question.lower() == 'exit':
#                 print("Exiting inference.")
#                 break
#             if not user_question.strip():
#                 print("Empty question. Please enter a valid question.")
#                 continue

#             # Preprocess the input question
#             question_tokens = preprocess_question(user_question, question_token_to_idx, Config.QUESTION_SEQ_LEN)
#             question_tensor = torch.tensor([question_tokens], dtype=torch.long).to(Config.DEVICE)  # (1, seq_len)
#             print(question_tokens)

#             # Perform inference
#             with torch.no_grad():
#                 predicted_program = model(question_tensor)  # (1, program_seq_len)
#                 predicted_program = predicted_program.cpu().numpy()[0]  # (program_seq_len,)
#             print(predicted_program)

#             # Decode the predicted program
#             predicted_program_tokens = decode_program(predicted_program, program_idx2word)
            
#             # Print the results
#             print("\n--- Inference Result ---")
#             print(f"Question: {user_question}")
#             print(f"Predicted Program: {predicted_program_tokens}\n")

# if __name__ == "__main__":
#     main()



import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import argparse
import json

# Configuration
class Config:
    VAL_QUESTIONS_H5 = '/Users/guoyuzhang/University/Y5/diss/vqa/code/h5_files/val_questions.h5'  # Update this path
    MODEL_PATH = 'models/best_seq2seq_program.pth'  # Path to the trained model
    VOCAB_PATH = '/Users/guoyuzhang/University/Y5/diss/vqa/code/data/vocab.json'  # Path to vocab.json
    BATCH_SIZE = 64
    EMBEDDING_DIM = 256
    LSTM_HIDDEN_DIM = 512
    PROGRAM_SEQ_LEN = 27  # Length of the program sequence
    QUESTION_SEQ_LEN = 46  # Length of the question sequence (as per training)
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    SEED = 42
    MAX_PRINT = 10  # Set to a number to limit printed examples, or None for all

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

# Model Definition with program_start_token_idx
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

    def forward(self, questions, program_targets=None):
        # Encode questions
        embedded = self.embedding(questions)  # (batch, seq_len, embedding_dim)
        encoder_out, (hidden, cell) = self.encoder(embedded)  # hidden: (1, batch, hidden_dim)
        
        if program_targets is not None:
            # Teacher forcing: Use actual program tokens as input
            embedded_program = self.embedding(program_targets)  # (batch, prog_seq_len, embedding_dim)
            decoder_out, _ = self.decoder(embedded_program, (hidden, cell))  # (batch, prog_seq_len, hidden_dim)
            program_output = self.fc(decoder_out)  # (batch, prog_seq_len, program_vocab_size)
            return program_output
        else:
            # Inference mode: Generate program tokens step-by-step
            batch_size = questions.size(0)
            generated_program = torch.zeros(batch_size, self.program_seq_len, dtype=torch.long).to(questions.device)
            input_token = torch.tensor([self.program_start_token_idx] * batch_size, dtype=torch.long).unsqueeze(1).to(questions.device)  # (batch, 1)

            hidden_dec = hidden
            cell_dec = cell

            for t in range(self.program_seq_len):
                embedded_token = self.embedding(input_token)  # (batch, 1, embedding_dim)
                decoder_out, (hidden_dec, cell_dec) = self.decoder(embedded_token, (hidden_dec, cell_dec))  # (batch, 1, hidden_dim)
                output = self.fc(decoder_out)  # (batch, 1, program_vocab_size)
                _, predicted = torch.max(output, dim=2)  # (batch, 1)
                generated_program[:, t] = predicted.squeeze(1)
                input_token = predicted  # Next input is current prediction

            return generated_program

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

# Evaluation Function without Teacher Forcing
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
            predicted_programs = model(questions)  # (batch, program_seq_len)

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

# Main Evaluation Loop
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate Seq2Seq Program Model')
    parser.add_argument('--val_h5', type=str, default=Config.VAL_QUESTIONS_H5,
                        help='Path to validation questions H5 file')
    parser.add_argument('--model_path', type=str, default=Config.MODEL_PATH,
                        help='Path to the trained model file')
    parser.add_argument('--vocab_path', type=str, default=Config.VOCAB_PATH,
                        help='Path to the vocabulary JSON file')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                        help='Batch size for evaluation')
    parser.add_argument('--embedding_dim', type=int, default=Config.EMBEDDING_DIM,
                        help='Embedding dimension size')
    parser.add_argument('--lstm_hidden_dim', type=int, default=Config.LSTM_HIDDEN_DIM,
                        help='LSTM hidden dimension size')
    parser.add_argument('--program_seq_len', type=int, default=Config.PROGRAM_SEQ_LEN,
                        help='Program sequence length')
    parser.add_argument('--max_print', type=int, default=Config.MAX_PRINT,
                        help='Maximum number of examples to print (set to None for all)')
    args = parser.parse_args()

    # Verify file paths
    if not os.path.exists(args.val_h5):
        raise FileNotFoundError(f"Validation H5 file not found at {args.val_h5}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at {args.model_path}")
    if not os.path.exists(args.vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {args.vocab_path}")

    # Load vocabulary
    try:
        (question_token_to_idx,
         question_idx2word,
         program_token_to_idx,
         program_idx2word) = load_vocab(args.vocab_path)
        print("Vocabulary loaded successfully.")
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        return

    # Get data info
    vocab_size, program_vocab_size = get_data_info(args.val_h5)
    print(f"Vocab Size: {vocab_size}, Program Vocab Size: {program_vocab_size}")

    # Create dataset indices
    with h5py.File(args.val_h5, 'r') as f:
        total_samples = f['questions'].shape[0]

    indices = list(range(total_samples))
    # If max_print is set, limit the indices
    if args.max_print is not None:
        indices = indices[:args.max_print]

    # For evaluation, use the entire validation set or limited by max_print
    val_indices = indices
    print(f"Validation samples: {len(val_indices)}")

    # Create dataset
    val_dataset = VQAProgramDataset(args.val_h5, val_indices)

    # Determine the number of CPU cores for DataLoader
    import multiprocessing
    num_workers = min(4, multiprocessing.cpu_count()) 

    # Create dataloader
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)

    # Device configuration
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Initialize model
    # Get program start token index from program_token_to_idx
    program_start_token_idx = program_token_to_idx.get("<START>", 1)  # Default to 1 if not found

    model = Seq2SeqModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        program_vocab_size=program_vocab_size,
        program_seq_len=args.program_seq_len,
        program_start_token_idx=program_start_token_idx
    ).to(device)

    # Load the trained model
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Define loss criterion (optional: skipped since loss isn't computed)
    # criterion = nn.CrossEntropyLoss()

    # Perform evaluation with printing
    val_acc_program, val_token_acc = evaluate(
        model, val_loader, device,
        question_idx2word=question_idx2word,
        program_idx2word=program_idx2word,
        max_print=args.max_print
    )
    print(f"\nValidation Acc Program: {val_acc_program:.4f}, "
          f"Validation Token Acc: {val_token_acc:.4f}")

if __name__ == "__main__":
    main()
