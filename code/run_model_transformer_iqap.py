import h5py
import torch
import torch.nn as nn
import math
import os
import json
import time
from sklearn.model_selection import train_test_split
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

# Configuration
class Config:
    LAPTOP_OR_CLUSTER = 'L'  # 'L' for Laptop, 'C' for Cluster
    PATH = '/exports/eddie/scratch/s1808795/vqa/code/' if LAPTOP_OR_CLUSTER == 'C' else '/Users/guoyuzhang/University/Y5/diss/vqa/code/'
    FEATURES_H5 = PATH + 'data/train_features.h5' if LAPTOP_OR_CLUSTER == 'C' else '/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5'
    QUESTIONS_H5 = PATH + 'h5_files/train_questions.h5'
    MODELS_DIR = PATH + 'models'
    MODEL_NAME = PATH + 'models/best_transformer_iqap.pth'  # Ensure this is the correct model path
    BATCH_SIZE = 64
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 256  # Match embedding_dim for consistency
    IMAGE_FEATURE_DIM = 1024  # Assuming image_feature_dim=1024
    NUM_CLASSES = None  # To be determined from data
    PROGRAM_SEQ_LEN = 27  # Updated to match training
    PROGRAM_VOCAB_SIZE = None  # To be determined from data
    MAX_QUESTION_LEN = 46  # Ensure this matches training
    MAX_PROGRAM_LEN = 27
    NUM_IMAGE_TOKENS = 14 * 14  # Spatial tokens
    SPECIAL_TOKEN_ID = 1  # Assuming 1 is <SOS>
    ANSWER_LOSS_WEIGHT = 1.0
    PROGRAM_LOSS_WEIGHT = 1.0
    ID_TO_TOKEN_PATH = PATH + 'data/vocab.json'  # Path to the token ID to token mapping
    VALIDATION_SPLIT = 0.005
    TEST_SPLIT = 0.99
    SEED = 42
    DEVICE = 'mps'  # 'mps' for Apple Silicon, 'cuda' for NVIDIA GPUs, 'cpu' otherwise

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

        if self.features_file is None:
            self.features_file = h5py.File(self.features_h5_path, 'r')
        if self.questions_file is None:
            self.questions_file = h5py.File(self.questions_h5_path, 'r')

        # Retrieve image index
        image_idx = self.questions_file['image_idxs'][actual_idx]
        image_features = self.features_file['features'][image_idx]  # Shape: (1024, 14, 14)
        # Treat image features as tokens: (1024, 14, 14) -> (196, 1024)
        image_features = torch.tensor(image_features, dtype=torch.float32).permute(1, 2, 0).contiguous().view(-1, Config.IMAGE_FEATURE_DIM)  # (196, 1024)

        # Retrieve question
        question = self.questions_file['questions'][actual_idx]  # Shape: (46,)
        question = torch.tensor(question, dtype=torch.long)

        # Retrieve answer
        answer = self.questions_file['answers'][actual_idx]
        answer = torch.tensor(answer, dtype=torch.long)

        # Retrieve program
        program = self.questions_file['programs'][actual_idx]  # Shape: (46,)  # Updated to match PROGRAM_SEQ_LEN=46
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

        # Question Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Special [CLS] token embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=0.1,
                                             max_len=num_image_tokens + Config.MAX_QUESTION_LEN + 1)  # +1 for [CLS]

        # Transformer Encoder with batch_first=True
        encoder_layers = TransformerEncoderLayer(d_model=embedding_dim, nhead=4, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=2)

        # Answer classifier (MLP)
        self.answer_classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, num_classes)
        )

        # Program Decoder using Transformer Decoder with batch_first=True
        self.program_decoder_embedding = nn.Embedding(program_vocab_size, embedding_dim, padding_idx=0)
        self.pos_decoder = PositionalEncoding(embedding_dim, dropout=0.1, max_len=program_seq_len + 1)  # +1 for <SOS>
        decoder_layers = TransformerDecoderLayer(d_model=embedding_dim, nhead=4, batch_first=True)
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

        # Apply positional encoding
        # Transformer with batch_first=True expects input of shape (batch, seq_len, embedding_dim)
        encoder_input = self.pos_encoder(encoder_input)  # (batch, seq_len, embedding_dim)

        # Pass through transformer encoder
        memory = self.transformer_encoder(encoder_input)  # (batch, seq_len, embedding_dim)

        # Extract [CLS] token's representation for answer classification
        cls_output = memory[:, 0, :]  # (batch, embedding_dim)

        # Answer Prediction
        answer_output = self.answer_classifier(cls_output)  # (batch, num_classes)

        # Program Decoding
        if program_targets is not None:
            # Training mode: Generate program logits without teacher forcing
            program_logits = self.autoregressive_program_generation(memory, Config.PROGRAM_SEQ_LEN)
            return answer_output, program_logits
        else:
            # Inference mode is handled separately
            return answer_output, None

    def autoregressive_program_generation(self, memory, program_seq_len):
        """
        Autoregressively generate program tokens during training without teacher forcing.

        Args:
            memory (Tensor): Encoder outputs (batch, seq_len, embedding_dim)
            program_seq_len (int): Length of the program sequence to generate

        Returns:
            program_logits (Tensor): (batch, program_seq_len, program_vocab_size)
        """
        batch_size = memory.size(0)
        device = memory.device

        # Initialize with <SOS> token
        sos_tokens = torch.full((batch_size, 1), Config.SPECIAL_TOKEN_ID, dtype=torch.long, device=device)  # Assuming <SOS> token id is 1
        generated_programs = sos_tokens  # (batch, 1)

        program_logits = []

        for _ in range(program_seq_len):
            # Embed the current program sequence
            program_embedded = self.program_decoder_embedding(generated_programs)  # (batch, seq_len, embedding_dim)

            # Apply positional encoding
            program_embedded = self.pos_decoder(program_embedded)  # (batch, seq_len, embedding_dim)

            # Generate target mask
            tgt_seq_len = program_embedded.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)  # (seq_len, seq_len)

            # Pass through transformer decoder
            decoder_output = self.transformer_decoder(
                tgt=program_embedded,
                memory=memory,
                tgt_mask=tgt_mask
            )  # (batch, seq_len, embedding_dim)

            # Output projection
            output_logits = self.program_output(decoder_output[:, -1, :])  # (batch, program_vocab_size)

            program_logits.append(output_logits.unsqueeze(1))  # (batch, 1, program_vocab_size)

            # Predict the next token
            _, next_tokens = torch.max(output_logits, dim=1)  # (batch,)

            # Append to the generated program
            generated_programs = torch.cat((generated_programs, next_tokens.unsqueeze(1)), dim=1)  # (batch, seq_len +1)

        # Concatenate logits along the sequence dimension
        program_logits = torch.cat(program_logits, dim=1)  # (batch, program_seq_len, program_vocab_size)

        return program_logits

# Decoding Function
def decode_program(program_ids, program_id_to_token):
    """
    Decode program token IDs to tokens.

    Args:
        program_ids (List[int]): List of program token IDs.
        program_id_to_token (dict): Mapping from token IDs to tokens.

    Returns:
        decoded_program (List[str]): List of program tokens.
    """
    return [program_id_to_token.get(int(id_), '<UNK>') for id_ in program_ids]

def decode_question(question_ids, question_id_to_token):
    """
    Decode question token IDs to tokens.

    Args:
        question_ids (List[int]): List of question token IDs.
        question_id_to_token (dict): Mapping from token IDs to tokens.

    Returns:
        decoded_question (List[str]): List of question tokens.
    """
    return [question_id_to_token.get(int(id_), '<UNK>') for id_ in question_ids]

def decode_answer(answer_id, answer_id_to_token):
    """
    Decode answer class ID to token.

    Args:
        answer_id (int): Answer class ID.
        answer_id_to_token (dict): Mapping from answer class IDs to tokens.

    Returns:
        decoded_answer (str): Answer token.
    """
    return answer_id_to_token.get(int(answer_id), '<UNK>')

# Inference Function
def infer(model, image_features, question, program_id_to_token, question_id_to_token, answer_id_to_token, device, max_program_length=Config.PROGRAM_SEQ_LEN):
    """
    Perform inference to generate answer and program.

    Args:
        model (VQAModel): Trained VQAModel.
        image_features (Tensor): (num_image_tokens, image_feature_dim)
        question (Tensor): (question_seq_len,)
        program_id_to_token (dict): Mapping from program token IDs to tokens.
        question_id_to_token (dict): Mapping from question token IDs to tokens.
        answer_id_to_token (dict): Mapping from answer class IDs to tokens.
        device (torch.device): Device to perform computation on.
        max_program_length (int): Maximum length of the program to generate.

    Returns:
        answer (int): Predicted answer class ID.
        program (List[int]): Generated program sequence.
        decoded_program (List[str]): Decoded program tokens.
        decoded_question (List[str]): Decoded question tokens.
        ground_truth_answer (str): Ground truth answer token.
        ground_truth_program (List[str]): Ground truth program tokens.
    """
    model.eval()
    with torch.no_grad():
        # Prepare inputs
        image_features = image_features.to(device).unsqueeze(0)  # (1, num_image_tokens, image_feature_dim)
        question = question.to(device).unsqueeze(0)  # (1, question_seq_len)

        # Forward pass for answer prediction
        answer_output, _ = model(image_features, question, program_targets=None)

        # Get predicted answer
        _, predicted_answer = torch.max(answer_output, 1)  # (1,)
        answer = predicted_answer.item()

        # Autoregressive Program Generation
        # Encode Image Features and Questions to get memory
        memory = model.transformer_encoder(model.pos_encoder(
            torch.cat((
                model.cls_token.expand(image_features.size(0), -1, -1),  # (batch, 1, embedding_dim)
                model.image_proj(image_features),  # (batch, num_image_tokens, embedding_dim)
                model.embedding(question)  # (batch, question_seq_len, embedding_dim)
            ), dim=1)  # (batch, 1 + num_image_tokens + question_seq_len, embedding_dim)
        ))  # (batch, seq_len, embedding_dim)

        # Initialize with <SOS> token
        sos_tokens = torch.full((1, 1), Config.SPECIAL_TOKEN_ID, dtype=torch.long, device=device)  # (1, 1)
        generated_programs = sos_tokens  # (1, 1)

        for _ in range(max_program_length):
            # Embed the current program sequence
            program_embedded = model.program_decoder_embedding(generated_programs)  # (1, seq_len, embedding_dim)

            # Apply positional encoding
            program_embedded = model.pos_decoder(program_embedded)  # (1, seq_len, embedding_dim)

            # Generate target mask
            tgt_seq_len = program_embedded.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)  # (seq_len, seq_len)

            # Pass through transformer decoder
            decoder_output = model.transformer_decoder(
                tgt=program_embedded,
                memory=memory,
                tgt_mask=tgt_mask
            )  # (1, seq_len, embedding_dim)

            # Output projection
            output_logits = model.program_output(decoder_output[:, -1, :])  # (1, program_vocab_size)

            # Predict the next token
            _, next_token = torch.max(output_logits, dim=1)  # (1,)

            # Append the predicted token
            generated_programs = torch.cat((generated_programs, next_token.unsqueeze(1)), dim=1)  # (1, seq_len +1)

        # Convert generated_programs to list and remove <SOS> token
        program = generated_programs.squeeze(0).tolist()  # (max_program_length +1,)
        if program and program[0] == Config.SPECIAL_TOKEN_ID:
            program = program[1:]  # Remove <SOS>

        # Decode program tokens
        decoded_program = decode_program(program, program_id_to_token)

    return answer, program, decoded_program

# Main Inference Function
def main():
    # Load Vocabulary Mapping
    if not os.path.exists(Config.ID_TO_TOKEN_PATH):
        print(f"Vocabulary file not found at {Config.ID_TO_TOKEN_PATH}")
        return

    with open(Config.ID_TO_TOKEN_PATH, 'r') as f:
        vocab = json.load(f)

    # Extract program_token_to_idx, question_token_to_idx, answer_token_to_idx
    program_token_to_idx = vocab.get('program_token_to_idx', {})
    question_token_to_idx = vocab.get('question_token_to_idx', {})
    answer_token_to_idx = vocab.get('answer_token_to_idx', {})

    # Create inverse mappings
    program_id_to_token = {int(v): k for k, v in program_token_to_idx.items()}
    question_id_to_token = {int(v): k for k, v in question_token_to_idx.items()}
    answer_id_to_token = {int(v): k for k, v in answer_token_to_idx.items()}

    # Determine vocab_size, num_classes, program_vocab_size
    vocab_size = len(question_token_to_idx)
    num_classes = len(answer_token_to_idx)
    program_vocab_size = len(program_token_to_idx)

    print(f"Vocab Size (Questions): {vocab_size}")
    print(f"Number of Classes (Answers): {num_classes}")
    print(f"Program Vocab Size: {program_vocab_size}")

    # Update Config with determined values
    Config.NUM_CLASSES = num_classes
    Config.PROGRAM_VOCAB_SIZE = program_vocab_size

    # Define indices for test split
    total_samples = 699989  # Ensure this matches your dataset
    indices = list(range(total_samples))

    # Split indices into train, val, test
    train_val_indices, test_indices = train_test_split(
        indices, test_size=Config.TEST_SPLIT, random_state=Config.SEED)
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=Config.VALIDATION_SPLIT / (1 - Config.TEST_SPLIT), random_state=Config.SEED)

    print(f"Total samples: {total_samples}")
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}, Test samples: {len(test_indices)}")

    # Create test dataset
    test_dataset = VQADataset(Config.FEATURES_H5, Config.QUESTIONS_H5, test_indices)

    # Determine the number of CPU cores for DataLoader
    import multiprocessing
    num_workers = min(4, multiprocessing.cpu_count())  # Adjust based on your CPU cores

    # Create DataLoader for test dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                                             num_workers=num_workers, pin_memory=True)

    # Instantiate the model
    device = torch.device(Config.DEVICE if torch.cuda.is_available() or torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = VQAModel(
        vocab_size=vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_classes=num_classes,
        program_vocab_size=program_vocab_size,
        program_seq_len=Config.PROGRAM_SEQ_LEN,
        num_image_tokens=Config.NUM_IMAGE_TOKENS,
        special_token_id=Config.SPECIAL_TOKEN_ID
    ).to(device)

    # Load the trained model state
    if not os.path.exists(Config.MODEL_NAME):
        print(f"Model file not found at {Config.MODEL_NAME}")
        return

    try:
        model.load_state_dict(torch.load(Config.MODEL_NAME, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except RuntimeError as e:
        print(f"RuntimeError while loading state_dict: {e}")
        return

    # Start inference
    start_time = time.time()

    # Initialize a list to store results
    results = []

    for batch_idx, (image_features, questions, answers, programs) in enumerate(test_loader):
        image_features = image_features.to(device)  # (batch, num_image_tokens, image_feature_dim)
        questions = questions.to(device)  # (batch, question_seq_len)

        # Iterate over the batch
        for i in range(image_features.size(0)):
            single_image_features = image_features[i]  # (num_image_tokens, image_feature_dim)
            single_question = questions[i]  # (question_seq_len,)
            ground_truth_answer = answers[i].item()
            ground_truth_program = programs[i].tolist()

            # Perform inference
            predicted_answer, generated_program, decoded_program = infer(
                model=model,
                image_features=single_image_features,
                question=single_question,
                program_id_to_token=program_id_to_token,
                question_id_to_token=question_id_to_token,
                answer_id_to_token=answer_id_to_token,
                device=device,
                max_program_length=Config.PROGRAM_SEQ_LEN
            )

            # Decode question and ground truth
            decoded_question = decode_question(single_question, question_id_to_token)  # Assuming ground_truth_program contains question tokens
            decoded_ground_truth_answer = decode_answer(ground_truth_answer, answer_id_to_token)
            decoded_ground_truth_program = decode_program(ground_truth_program, program_id_to_token)

            # Append to results
            sample_number = batch_idx * Config.BATCH_SIZE + i + 1
            results.append({
                'sample_id': sample_number,
                'question': decoded_question,
                'ground_truth_answer': decoded_ground_truth_answer,
                'predicted_answer': predicted_answer,
                'ground_truth_program': decoded_ground_truth_program,
                'generated_program': decoded_program
            })

            # Print the results
            print(f"\nSample {sample_number}:")
            # print(single_question)
            print(f"Question: {' '.join(decoded_question)}")
            print(f"Ground Truth Answer: {decoded_ground_truth_answer}")
            print(f"Predicted Answer Class ID: {predicted_answer}")
            print(f"Ground Truth Program Token IDs: {ground_truth_program}")
            print(f"Generated Program Token IDs: {generated_program}")
            print(f"Ground Truth Program: {' '.join(decoded_ground_truth_program)}")
            print(f"Generated Program: {' '.join(decoded_program)}")
            
        

    end_time = time.time()
    print(f"\nInference completed in {end_time - start_time:.2f} seconds.")

    # Optionally, save results to a JSON file
    output_file = os.path.join(Config.MODELS_DIR, 'inference_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Inference results saved to {output_file}")

if __name__ == "__main__":
    main()
