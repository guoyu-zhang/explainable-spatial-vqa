import h5py
import torch
import torch.nn as nn
import math
import os
import numpy as np

# Configuration
class Config:
    LAPTOP_OR_CLUSTER = 'L'  # Change this depending on running on cluster or PC
    PATH = '/exports/eddie/scratch/s1808795/vqa/code/' if LAPTOP_OR_CLUSTER == 'CLUSTER' else '/Users/guoyuzhang/University/Y5/diss/vqa/code/'
    FEATURES_H5 = PATH + 'data/train_features.h5' if LAPTOP_OR_CLUSTER == 'CLUSTER' else '/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5'
    QUESTIONS_H5 = PATH + 'h5_files/train_questions.h5'
    MODELS_DIR = PATH + 'models'
    MODEL_NAME = PATH + 'models/best_transformer_iqap.pth'
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 256  # Match embedding_dim for consistency
    IMAGE_FEATURE_DIM = 1024  # Assuming image_feature_dim=1024
    NUM_CLASSES = None  # To be loaded from data
    PROGRAM_SEQ_LEN = 27  # Length of the program sequence
    PROGRAM_VOCAB_SIZE = None  # To be loaded from data
    MAX_QUESTION_LEN = 46  # As per existing data
    NUM_IMAGE_TOKENS = 14 * 14  # Spatial tokens
    SPECIAL_TOKEN_ID = 1  # Assuming 1 is <SOS>

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

# Custom Dataset for single sample inference
class VQADatasetSingleSample:
    def __init__(self, features_h5_path, questions_h5_path, sample_idx):
        self.features_h5_path = features_h5_path
        self.questions_h5_path = questions_h5_path
        self.sample_idx = sample_idx
        self.features_file = h5py.File(self.features_h5_path, 'r')
        self.questions_file = h5py.File(self.questions_h5_path, 'r')

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        actual_idx = self.sample_idx

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
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)

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
        decoder_layers = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=4)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=2)
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

        # Program Decoder (Inference Mode)
        if program_targets is None:
            program_logits = self.autoregressive_program_generation(memory, Config.PROGRAM_SEQ_LEN)
            return answer_output, program_logits
        else:
            # During training, return program logits
            program_logits = self.autoregressive_program_generation(memory, Config.PROGRAM_SEQ_LEN)
            return answer_output, program_logits

    def autoregressive_program_generation(self, memory, program_seq_len):
        """
        Autoregressively generate program tokens during inference.

        Args:
            memory (Tensor): Encoder outputs (seq_len, batch, embedding_dim)
            program_seq_len (int): Length of the program sequence to generate

        Returns:
            generated_programs (Tensor): (batch, program_seq_len)
        """
        batch_size = memory.size(1)
        device = memory.device

        # Initialize with <SOS> token
        sos_tokens = torch.full((batch_size, 1), Config.SPECIAL_TOKEN_ID, dtype=torch.long, device=device)  # Assuming <SOS> token id is 1
        generated_programs = sos_tokens  # (batch, 1)

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

            # Predict the next token
            _, next_tokens = torch.max(output_logits, dim=1)  # (batch,)

            # Append to the generated program
            generated_programs = torch.cat((generated_programs, next_tokens.unsqueeze(1)), dim=1)  # (batch, seq_len +1)

        # Remove the <SOS> token
        generated_programs = generated_programs[:, 1:]  # (batch, program_seq_len)

        return generated_programs

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

# Utility function to load the best model
def load_model(device):
    # Get data info
    vocab_size, num_classes, program_vocab_size = get_data_info(Config.QUESTIONS_H5)
    Config.NUM_CLASSES = num_classes
    Config.PROGRAM_VOCAB_SIZE = program_vocab_size

    # Initialize model
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

    # Load the trained model weights
    model.load_state_dict(torch.load(Config.MODEL_NAME, map_location=device))
    model.eval()
    return model

# Inference Function
def run_inference(sample_idx=0):
    # Device configuration
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Load the model
    model = load_model(device)
    print("Model loaded successfully.")

    # Create dataset
    dataset = VQADatasetSingleSample(Config.FEATURES_H5, Config.QUESTIONS_H5, sample_idx)

    # Get the sample
    image_features, question, answer, program = dataset[0]
    image_features = image_features.unsqueeze(0).to(device)  # (1, num_image_tokens, image_feature_dim)
    question = question.unsqueeze(0).to(device)  # (1, question_seq_len)

    # Run inference
    with torch.no_grad():
        answer_output, generated_program = model(image_features, question)

        # Get predicted answer
        _, predicted_answer = torch.max(answer_output, 1)  # (1,)
        predicted_answer = predicted_answer.item()

        # Get generated program
        generated_program = generated_program.squeeze(0).tolist()  # (program_seq_len,)

    # Load ground truth for reference
    with h5py.File(Config.QUESTIONS_H5, 'r') as f:
        ground_truth_answer = int(f['answers'][sample_idx])
        ground_truth_program = f['programs'][sample_idx].tolist()

    # Print the results
    print(f"\n=== Inference Results for Sample Index: {sample_idx} ===")
    print(f"Question: {question_to_string(question.squeeze(0).tolist())}")
    print(f"Predicted Answer: {predicted_answer} | Ground Truth Answer: {ground_truth_answer}")
    print(f"Predicted Program: {generated_program}")
    print(f"Ground Truth Program: {ground_truth_program}")
    print("=============================================\n")

# Utility function to convert question indices to string (if vocabulary is available)
def question_to_string(question_indices):
    # Placeholder function: Replace with actual vocabulary mapping if available
    # For example, load a vocabulary file and map indices to words
    # Here, we'll just return the indices as a string
    return ' '.join(map(str, question_indices))

# Utility function to convert answer index to string (if label mapping is available)
def answer_to_string(answer_index):
    # Placeholder function: Replace with actual label mapping if available
    # For example, load a label file and map indices to labels
    # Here, we'll just return the index as a string
    return str(answer_index)

# Utility function to convert program indices to string (if program vocabulary is available)
def program_to_string(program_indices):
    # Placeholder function: Replace with actual program vocabulary mapping if available
    # For example, load a program vocabulary file and map indices to instructions
    # Here, we'll just return the indices as a string
    return ' '.join(map(str, program_indices))

if __name__ == "__main__":
    # Specify the sample index you want to infer
    SAMPLE_INDEX = 60  # Change this to select a different sample

    run_inference(sample_idx=SAMPLE_INDEX)
