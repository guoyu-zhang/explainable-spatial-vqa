import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math
import os
import copy

# Import transformer modules
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

# Configuration
class Config:
    LAPTOP_OR_CLUSTER = 'CLUSTER'  # Change this depending on running on cluster or PC
    PATH = '/exports/eddie/scratch/s1808795/vqa/code/' if LAPTOP_OR_CLUSTER == 'CLUSTER' else '/Users/guoyuzhang/University/Y5/diss/vqa/code/'
    FEATURES_H5 = PATH + 'data/train_features.h5' if LAPTOP_OR_CLUSTER == 'CLUSTER' else '/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5'
    QUESTIONS_H5 = PATH + 'h5_files/train_questions.h5'
    MODELS_DIR = PATH + 'models'
    MODEL_NAME = PATH + 'models/best_transformer_iqap.pth'
    BATCH_SIZE = 64
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 256  # Match embedding_dim for consistency
    IMAGE_FEATURE_DIM = 1024  # Assuming image_feature_dim=1024
    NUM_CLASSES = None  # To be determined from data
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    VALIDATION_SPLIT = 0.1
    TEST_SPLIT = 0.1
    SEED = 42
    PROGRAM_SEQ_LEN = 27  # Length of the program sequence
    PROGRAM_VOCAB_SIZE = None  # To be determined from data
    MAX_QUESTION_LEN = 46  # As per existing data
    MAX_PROGRAM_LEN = 27
    NUM_IMAGE_TOKENS = 14 * 14  # Spatial tokens
    SPECIAL_TOKEN_ID = 1  # Assuming 1 is <SOS>
    ANSWER_LOSS_WEIGHT = 1.0
    PROGRAM_LOSS_WEIGHT = 1.0

# Set random seeds and deterministic behavior
torch.manual_seed(Config.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        # position: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # div_term: (d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe: (max_len, d_model)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (seq_len, batch_size, embedding_dim)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Utility function to generate square subsequent mask
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

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
        # Treat image features as tokens: (1024, 14, 14) -> (196, 1024)
        image_features = torch.tensor(image_features, dtype=torch.float32).permute(1, 2, 0).contiguous().view(-1, 1024)  # (196, 1024)

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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 program_vocab_size, program_seq_len, num_image_tokens, special_token_id=1):
        super(VQAModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_image_tokens = num_image_tokens
        self.special_token_id = special_token_id

        # Image feature projection
        self.image_proj = nn.Linear(Config.IMAGE_FEATURE_DIM, embedding_dim)  # Assuming image_feature_dim=1024

        # Question Encoder using Transformer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Special [CLS] token embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=0.1,
                                             max_len=num_image_tokens + Config.MAX_QUESTION_LEN + 1)  # +1 for [CLS] token

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model=embedding_dim, nhead=4)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=1)

        # Answer classifier (MLP)
        self.answer_classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

        # Program Decoder using Transformer Decoder
        self.program_decoder_embedding = nn.Embedding(program_vocab_size, embedding_dim, padding_idx=0)
        self.pos_decoder = PositionalEncoding(embedding_dim, dropout=0.1, max_len=Config.PROGRAM_SEQ_LEN + 1)  # +1 for <SOS>
        decoder_layers = TransformerDecoderLayer(d_model=embedding_dim, nhead=4)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers=2)
        self.program_output = nn.Linear(embedding_dim, program_vocab_size)

    def forward(self, image_features, questions, program_targets=None, max_program_length=None):
        """
        Args:
            image_features (Tensor): (batch, num_image_tokens, image_feature_dim)
            questions (Tensor): (batch, question_seq_len)
            program_targets (Tensor, optional): (batch, program_seq_len)
            max_program_length (int, optional): Maximum length of program to generate during inference

        Returns:
            answer_output (Tensor): (batch, num_classes)
            program_logits (Tensor): (batch, program_seq_len, program_vocab_size) during training
        """
        batch_size = image_features.size(0)

        # Encode Image Features as tokens
        # image_features: (batch, num_image_tokens, image_feature_dim)
        image_encoded = self.image_proj(image_features)  # (batch, num_image_tokens, embedding_dim)

        # Encode Questions
        # questions: (batch, question_seq_len)
        question_embedded = self.embedding(questions)  # (batch, question_seq_len, embedding_dim)

        # Concatenate [CLS] token, image tokens, and question tokens
        # Create [CLS] token for each batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, embedding_dim)

        # Concatenate
        # sequence: [CLS] + image_tokens + question_tokens
        encoder_input = torch.cat((cls_tokens, image_encoded, question_embedded), dim=1)  # (batch, 1 + num_image_tokens + question_seq_len, embedding_dim)

        # Permute for transformer (seq_len, batch, embedding_dim)
        encoder_input = encoder_input.permute(1, 0, 2)  # (seq_len, batch, embedding_dim)

        # Apply positional encoding
        encoder_input = self.pos_encoder(encoder_input)  # (seq_len, batch, embedding_dim)

        # Pass through transformer encoder
        memory = self.transformer_encoder(encoder_input)  # (seq_len, batch, embedding_dim)

        # Extract [CLS] token's representation for answer classification
        cls_output = memory[0, :, :]  # (batch, embedding_dim)

        # Answer prediction via MLP
        answer_output = self.answer_classifier(cls_output)  # (batch, num_classes)


        # Training mode: Autoregressive decoding without teacher forcing
        program_logits = self.autoregressive_program_generation(memory, Config.PROGRAM_SEQ_LEN)
        return answer_output, program_logits


    def autoregressive_program_generation(self, memory, program_seq_len):
        """
        Autoregressively generate program tokens during training without teacher forcing.

        Args:
            memory (Tensor): Encoder outputs (seq_len, batch, embedding_dim)
            program_seq_len (int): Length of the program sequence to generate

        Returns:
            program_logits (Tensor): (batch, program_seq_len, program_vocab_size)
        """
        batch_size = memory.size(1)
        device = memory.device

        # Initialize with <SOS> token
        sos_tokens = torch.full((batch_size, 1), Config.SPECIAL_TOKEN_ID, dtype=torch.long, device=device)  # Assuming <SOS> token id is 1
        generated_programs = sos_tokens  # (batch, 1)

        program_logits = []

        for _ in range(program_seq_len):
            # Embed the current program sequence
            program_embedded = self.program_decoder_embedding(generated_programs)  # (batch, seq_len, embedding_dim)

            # Permute for transformer decoder (seq_len, batch, embedding_dim)
            program_embedded = program_embedded.permute(1, 0, 2)  # (seq_len, batch, embedding_dim)

            # Apply positional encoding
            program_embedded = self.pos_decoder(program_embedded)  # (seq_len, batch, embedding_dim)

            # Generate target mask
            tgt_seq_len = program_embedded.size(0)
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)  # (seq_len, seq_len)

            # Pass through transformer decoder
            decoder_output = self.transformer_decoder(
                tgt=program_embedded,
                memory=memory,
                tgt_mask=tgt_mask
            )  # (seq_len, batch, embedding_dim)

            # Output projection
            output_logits = self.program_output(decoder_output[-1, :, :])  # (batch, program_vocab_size)

            program_logits.append(output_logits.unsqueeze(1))  # Append along time dimension

            # Predict the next token
            _, next_tokens = torch.max(output_logits, dim=1)  # (batch,)

            # Append to the generated program
            generated_programs = torch.cat((generated_programs, next_tokens.unsqueeze(1)), dim=1)  # (batch, seq_len +1)

        # Concatenate logits along the sequence dimension
        program_logits = torch.cat(program_logits, dim=1)  # (batch, program_seq_len, program_vocab_size)

        return program_logits

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
        image_features = image_features.to(device)  # (batch, num_image_tokens, image_feature_dim)
        questions = questions.to(device)  # (batch, question_seq_len)
        answers = answers.to(device)  # (batch,)
        programs = programs.to(device)  # (batch, program_seq_len)

        optimizer.zero_grad()
        outputs_answer, outputs_program = model(image_features, questions, program_targets=programs)

        # Compute answer loss
        loss_answer = criterion_answer(outputs_answer, answers)

        # Compute program loss
        # outputs_program: (batch, program_seq_len, program_vocab_size)
        # programs: (batch, program_seq_len)
        program_output_targets = programs  # Assuming programs include <SOS>

        # Flatten the outputs and targets for loss computation
        program_logits = outputs_program.reshape(-1, outputs_program.size(-1))  # (batch * program_seq_len, program_vocab_size)
        program_targets_flat = program_output_targets.reshape(-1)  # (batch * program_seq_len)

        # Compute program loss
        loss_program = criterion_program(program_logits, program_targets_flat)

        # Total loss with weighting
        loss = Config.ANSWER_LOSS_WEIGHT * loss_answer + Config.PROGRAM_LOSS_WEIGHT * loss_program
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * image_features.size(0)

        # Compute answer accuracy
        _, predicted_answer = torch.max(outputs_answer, 1)
        correct_answer += (predicted_answer == answers).sum().item()

        # Compute program accuracy (exact match)
        _, predicted_program = torch.max(outputs_program, 2)  # (batch, program_seq_len)
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

    # Lists to store sample predictions and ground truths
    sample_predictions = []
    sample_ground_truths = []

    with torch.no_grad():
        for image_features, questions, answers, programs in tqdm(dataloader, desc="Evaluating", leave=False):
            image_features = image_features.to(device)
            questions = questions.to(device)
            answers = answers.to(device)
            programs = programs.to(device)

            outputs_answer, outputs_program = model(image_features, questions, program_targets=programs)

            # Compute answer loss
            loss_answer = criterion_answer(outputs_answer, answers)

            # Compute program loss
            program_output_targets = programs  # Assuming programs include <SOS>
            program_logits = outputs_program.reshape(-1, outputs_program.size(-1))  # (batch * program_seq_len, program_vocab_size)
            program_targets_flat = program_output_targets.reshape(-1)  # (batch * program_seq_len)
            loss_program = criterion_program(program_logits, program_targets_flat)

            # Total loss with weighting
            loss = Config.ANSWER_LOSS_WEIGHT * loss_answer + Config.PROGRAM_LOSS_WEIGHT * loss_program
            running_loss += loss.item() * image_features.size(0)

            # Compute answer accuracy
            _, predicted_answer = torch.max(outputs_answer, 1)
            correct_answer += (predicted_answer == answers).sum().item()

            # Compute program accuracy (exact match)
            _, predicted_program = torch.max(outputs_program, 2)  # (batch, program_seq_len)
            exact_matches = (predicted_program == programs).all(dim=1).sum().item()
            correct_program += exact_matches

            # Compute token-wise accuracy
            correct_tokens += (predicted_program == programs).sum().item()
            total_tokens += programs.numel()

            total += answers.size(0)

            # **Modification Starts Here**
            # Store predicted and ground truth answers and programs
            # for i in range(answers.size(0)):
            #     predicted_ans = predicted_answer[i].item()
            #     ground_truth_ans = answers[i].item()
            #     predicted_prog = predicted_program[i].tolist()
            #     ground_truth_prog = programs[i].tolist()

            #     sample_predictions.append({
            #         'predicted_answer': predicted_ans,
            #         'ground_truth_answer': ground_truth_ans,
            #         'predicted_program': predicted_prog,
            #         'ground_truth_program': ground_truth_prog
            #     })
            # **Modification Ends Here**

    epoch_loss = running_loss / total
    epoch_acc_answer = correct_answer / total
    epoch_acc_program = correct_program / total
    epoch_token_acc = correct_tokens / total_tokens

    # **Addition Starts Here**
    # Print all sample predictions and ground truths
    # print("\nValidation Predictions:")
    # for idx, sample in enumerate(sample_predictions, 1):
    #     print(f"Sample {idx}:")
    #     print(f"  Predicted Answer: {sample['predicted_answer']}, Ground Truth Answer: {sample['ground_truth_answer']}")
    #     print(f"  Predicted Program: {sample['predicted_program']}")
    #     print(f"  Ground Truth Program: {sample['ground_truth_program']}\n")
    # **Addition Ends Here**

    return epoch_loss, epoch_acc_answer, epoch_acc_program, epoch_token_acc

# Main Training Loop
def main():
    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    # Get data info
    vocab_size, num_classes, program_vocab_size = get_data_info(Config.QUESTIONS_H5)
    Config.NUM_CLASSES = num_classes
    Config.PROGRAM_VOCAB_SIZE = program_vocab_size
    print(f"Vocab Size: {vocab_size}, Number of Classes: {num_classes}, Program Vocab Size: {program_vocab_size}")

    # Create dataset indices
    total_samples = 699989  # Reduced to 3,500 samples as per your scenario
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
    num_workers = min(4, multiprocessing.cpu_count())  # Adjust based on your CPU cores

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
        hidden_dim=Config.HIDDEN_DIM,
        num_classes=Config.NUM_CLASSES,
        program_vocab_size=Config.PROGRAM_VOCAB_SIZE,
        program_seq_len=Config.PROGRAM_SEQ_LEN,
        num_image_tokens=Config.NUM_IMAGE_TOKENS,
        special_token_id=Config.SPECIAL_TOKEN_ID
    ).to(device)

    criterion_answer = nn.CrossEntropyLoss()
    criterion_program = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is padding index
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())
    patience = 10
    trigger_times = 0

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

        # Early Stopping based on validation answer accuracy
        if val_acc_answer > best_val_acc:
            best_val_acc = val_acc_answer
            best_model_state = copy.deepcopy(model.state_dict())
            trigger_times = 0
            torch.save(model.state_dict(), Config.MODEL_NAME)
            print("Best model saved.")
        else:
            trigger_times += 1
            print(f"No improvement in validation answer accuracy for {trigger_times} epochs.")
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

        # Step the scheduler
        scheduler.step()

    # Load the best model for testing
    model.load_state_dict(best_model_state)
    test_loss, test_acc_answer, test_acc_program, test_token_acc = evaluate(
        model, test_loader, criterion_answer, criterion_program, device)
    print(f"\nTest Loss: {test_loss:.4f}, "
          f"Test Acc Answer: {test_acc_answer:.4f}, "
          f"Test Acc Program: {test_acc_program:.4f}, "
          f"Test Token Acc: {test_token_acc:.4f}")

if __name__ == "__main__":
    main()
