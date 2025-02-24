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
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer
)

###############################################################################
#                              CONFIGURATION                                  #
###############################################################################
class Config:
    LAPTOP_OR_CLUSTER = 'L'  # 'CLUSTER' if on HPC, else 'L'
    PATH = (
        '/exports/eddie/scratch/s1808795/vqa/code/' 
        if LAPTOP_OR_CLUSTER == 'CLUSTER' 
        else '/Users/guoyuzhang/University/Y5/diss/vqa/code/'
    )

    FEATURES_H5 = (
        PATH + 'data/train_features.h5'
        if LAPTOP_OR_CLUSTER == 'CLUSTER'
        else '/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5'
    )
    QUESTIONS_H5 = PATH + 'h5_files/train_questions.h5'
    BBOX_H5      = PATH + 'h5_files/train_scenes.h5'  # bounding boxes shaped (N, 10, 4)
    MODELS_DIR   = PATH + 'models'
    MODEL_NAME   = PATH + 'models/bb_best_transformer_iqap.pth'

    BATCH_SIZE       = 64
    EMBEDDING_DIM    = 256
    HIDDEN_DIM       = 256
    IMAGE_FEATURE_DIM= 1024
    NUM_CLASSES      = None
    NUM_EPOCHS       = 10   # For demonstration
    LEARNING_RATE    = 1e-3
    VALIDATION_SPLIT = 0.005
    TEST_SPLIT       = 0.99
    SEED = 42

    PROGRAM_SEQ_LEN = 27
    PROGRAM_VOCAB_SIZE = None

    MAX_QUESTION_LEN = 46
    NUM_IMAGE_TOKENS = 14 * 14  # 196
    SPECIAL_TOKEN_ID = 1  # <SOS> token ID

    # Combined seq = program(27) + answer(1) => 28
    COMBINED_SEQ_LEN = PROGRAM_SEQ_LEN + 1

###############################################################################
#                           RANDOM SEED SETTINGS                              #
###############################################################################
torch.manual_seed(Config.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###############################################################################
#                          POSITIONAL ENCODING                                #
###############################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len,1,d_model)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (seq_len, batch_size, embedding_dim)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

###############################################################################
#                 UTILITY: Generate Square Subsequent Mask                   #
###############################################################################
def generate_square_subsequent_mask(sz):
    """
    Creates an upper-triangular matrix of -inf, 0 for masking. 
    Used for autoregressive decoding.
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

###############################################################################
#                     BOUNDING BOX / IOU UTILITIES                            #
###############################################################################
def bbox_iou_2d(pred_box, gt_box):
    """
    pred_box, gt_box: [x_min, y_min, x_max, y_max], all in [0,1].
    Returns scalar IoU in [0,1].
    """
    ixmin = max(pred_box[0], gt_box[0])
    iymin = max(pred_box[1], gt_box[1])
    ixmax = min(pred_box[2], gt_box[2])
    iymax = min(pred_box[3], gt_box[3])

    iw = max(ixmax - ixmin, 0.0)
    ih = max(iymax - iymin, 0.0)
    intersection = iw * ih

    pred_area = max(pred_box[2] - pred_box[0], 0.0) * max(pred_box[3] - pred_box[1], 0.0)
    gt_area   = max(gt_box[2]  - gt_box[0], 0.0)   * max(gt_box[3]  - gt_box[1], 0.0)

    union = pred_area + gt_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union

def batch_mean_iou(bbox_preds, bbox_gts):
    """
    bbox_preds, bbox_gts: shape (batch, 10, 4)
      - Some images may have <10 real boxes, the rest are zeros => skip them.

    Returns average IoU over all non-zero GT boxes in the batch.
    """
    batch_size = bbox_preds.size(0)
    ious = []
    for i in range(batch_size):
        for j in range(10):
            gt_box = bbox_gts[i, j, :].tolist()
            # If GT is all zeros, skip
            if all(abs(x) < 1e-8 for x in gt_box):
                continue
            # Pred box
            pred_box = bbox_preds[i, j, :].tolist()
            # Optionally clamp predictions to [0,1]
            pred_box = [min(max(coord, 0.0), 1.0) for coord in pred_box]

            iou_val = bbox_iou_2d(pred_box, gt_box)
            ious.append(iou_val)
    if len(ious) == 0:
        return 0.0
    return float(np.mean(ious))

###############################################################################
#                               DATASET                                       #
###############################################################################
class VQADataset(Dataset):
    """
    Returns:
      image_features: (196,1024)
      question:       (46,)
      combined_seq:   (28,) => 27 program tokens + 1 answer
      bboxes:         (10,4) => up to 10 boxes, zero-padded if fewer
    """
    def __init__(self, features_h5_path, questions_h5_path, bboxes_h5_path, indices):
        self.features_h5_path = features_h5_path
        self.questions_h5_path = questions_h5_path
        self.bboxes_h5_path    = bboxes_h5_path
        self.indices = indices

        self.features_file  = None
        self.questions_file = None
        self.bboxes_file    = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        # Lazy open HDF5
        if self.features_file is None:
            self.features_file = h5py.File(self.features_h5_path, 'r')
        if self.questions_file is None:
            self.questions_file = h5py.File(self.questions_h5_path, 'r')
        if self.bboxes_file is None:
            self.bboxes_file = h5py.File(self.bboxes_h5_path, 'r')

        # 1) Retrieve image index
        image_idx = self.questions_file['image_idxs'][actual_idx]

        # 2) Load image features (1024,14,14) -> (196,1024)
        image_feats = self.features_file['features'][image_idx]
        image_feats = torch.tensor(image_feats, dtype=torch.float32).permute(1,2,0).reshape(-1,1024)

        # 3) Question
        question = torch.tensor(self.questions_file['questions'][actual_idx], dtype=torch.long)

        # 4) Program+Answer => combined_seq
        program = torch.tensor(self.questions_file['programs'][actual_idx], dtype=torch.long)  # (27,)
        ans_val = torch.tensor([self.questions_file['answers'][actual_idx]], dtype=torch.long)  # (1,)
        combined_seq = torch.cat([program, ans_val], dim=0)  # (28,)

        # 5) BBoxes => (10,4)
        bboxes = self.bboxes_file['bounding_boxes'][image_idx]
        bboxes = torch.tensor(bboxes, dtype=torch.float32)

        return image_feats, question, combined_seq, bboxes

    def __del__(self):
        if self.features_file is not None:
            self.features_file.close()
        if self.questions_file is not None:
            self.questions_file.close()
        if self.bboxes_file is not None:
            self.bboxes_file.close()

###############################################################################
#                            MODEL DEFINITION                                 #
###############################################################################
class VQAModel(nn.Module):
    """
    1) Outputs program+answer (28 tokens)
    2) Outputs 10 bounding boxes => (batch,10,4)
    """
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        program_vocab_size,
        program_seq_len,
        num_image_tokens,
        special_token_id=1
    ):
        super(VQAModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_image_tokens = num_image_tokens
        self.special_token_id = special_token_id

        # We'll generate program_seq_len+1=28 tokens
        self.seq_len = program_seq_len + 1

        # (A) Image feature projection
        self.image_proj = nn.Linear(Config.IMAGE_FEATURE_DIM, embedding_dim)

        # (B) Question embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # (C) [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1,1,embedding_dim))

        # (D) Positional encoding
        self.pos_encoder = PositionalEncoding(
            embedding_dim,
            dropout=0.1,
            max_len=num_image_tokens + Config.MAX_QUESTION_LEN + 1
        )

        # (E) Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model=embedding_dim, nhead=4)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=1)

        # (F) Program+Answer Decoder
        self.decoder_embedding = nn.Embedding(program_vocab_size, embedding_dim, padding_idx=0)
        self.pos_decoder = PositionalEncoding(embedding_dim, dropout=0.1, max_len=self.seq_len + 1)
        decoder_layers = TransformerDecoderLayer(d_model=embedding_dim, nhead=4)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers=1)
        self.output_layer = nn.Linear(embedding_dim, program_vocab_size)

        # (G) BBox Regressor => always predict 10 boxes
        # We'll pool the image tokens from the encoder => shape(196,batch,emb_dim)
        self.bbox_regressor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 40)  # 10 * 4
        )

    def forward(self, image_features, questions):
        """
        Returns:
          seq_logits: (batch,28,program_vocab_size)
          bbox_preds: (batch,10,4)
        """
        batch_size = image_features.size(0)

        # 1) Encode image
        img_enc = self.image_proj(image_features)  # (batch,196,emb_dim)

        # 2) Encode question
        q_emb = self.embedding(questions)          # (batch,question_len,emb_dim)

        # 3) [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch,1,emb_dim)

        # 4) Combine => (batch, 1 + 196 + question_len, emb_dim)
        encoder_input = torch.cat((cls_tokens, img_enc, q_emb), dim=1)

        # 5) Permute => (seq_len,batch,emb_dim)
        encoder_input = encoder_input.permute(1,0,2)

        # 6) Transformer encoder
        encoder_input = self.pos_encoder(encoder_input)
        memory = self.transformer_encoder(encoder_input)  # (seq_len,batch,emb_dim)

        # 7) BBox from image tokens => [1 : 1+196], then pool
        image_tokens = memory[1:1+Config.NUM_IMAGE_TOKENS, :, :]  # (196,batch,emb_dim)
        image_tokens_pooled = image_tokens.mean(dim=0)            # (batch,emb_dim)

        bbox_preds = self.bbox_regressor(image_tokens_pooled)     # (batch,40)
        bbox_preds = bbox_preds.view(batch_size, 10, 4)           # (batch,10,4)

        # 8) Autoregressive decode for program+answer
        seq_logits = self.autoregressive_decode(memory, self.seq_len)
        return seq_logits, bbox_preds

    def autoregressive_decode(self, memory, seq_len):
        """
        Generate the entire combined sequence of length seq_len=28.
        """
        batch_size = memory.size(1)
        device = memory.device

        # Start with <SOS>
        sos_tokens = torch.full(
            (batch_size, 1),
            Config.SPECIAL_TOKEN_ID,
            dtype=torch.long,
            device=device
        )
        generated_seq = sos_tokens
        all_logits = []

        for _ in range(seq_len):
            embedded_seq = self.decoder_embedding(generated_seq)  # (batch, cur_len, emb_dim)
            embedded_seq = embedded_seq.permute(1,0,2)
            embedded_seq = self.pos_decoder(embedded_seq)

            tgt_seq_len = embedded_seq.size(0)
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)

            decoder_output = self.transformer_decoder(
                tgt=embedded_seq,
                memory=memory,
                tgt_mask=tgt_mask
            )  # (cur_len,batch,emb_dim)

            output_logits = self.output_layer(decoder_output[-1, :, :])  # (batch,vocab_size)
            all_logits.append(output_logits.unsqueeze(1))

            # Pick next token
            _, next_tok = torch.max(output_logits, dim=1)
            next_tok = next_tok.unsqueeze(1)
            generated_seq = torch.cat([generated_seq, next_tok], dim=1)

        seq_logits = torch.cat(all_logits, dim=1)  # (batch, seq_len, vocab_size)
        return seq_logits

###############################################################################
#                        GET DATA INFO / VOCAB SIZES                          #
###############################################################################
def get_data_info(questions_h5_path):
    with h5py.File(questions_h5_path, 'r') as f:
        questions = f['questions']
        answers   = f['answers']
        programs  = f['programs']

        vocab_size = int(np.max(questions)) + 1
        program_vocab_size = max(int(np.max(programs)), int(np.max(answers))) + 1
    return vocab_size, program_vocab_size

###############################################################################
#                          TRAINING FUNCTION                                  #
###############################################################################
def train_epoch(model, dataloader, criterion_seq, criterion_bbox, optimizer, device):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0
    total = 0

    for image_features, questions, combined_seq, bboxes_gt in tqdm(dataloader, desc="Training", leave=False):
        image_features = image_features.to(device)
        questions     = questions.to(device)
        combined_seq  = combined_seq.to(device)
        bboxes_gt     = bboxes_gt.to(device)

        optimizer.zero_grad()
        seq_logits, bbox_preds = model(image_features, questions)

        # 1. Program+Answer Loss
        seq_logits_flat = seq_logits.reshape(-1, seq_logits.size(-1))  # (batch*28,vocab_size)
        targets_flat    = combined_seq.reshape(-1)
        loss_seq = criterion_seq(seq_logits_flat, targets_flat)

        # 2. Bounding Box Loss (Smooth L1 Loss)
        # Create a mask for non-zero ground truth boxes
        mask = (bboxes_gt.sum(dim=2, keepdim=True) > 0)  # (batch,10,1)

        # Compute Smooth L1 Loss only on valid boxes
        loss_bbox = criterion_bbox(bbox_preds, bboxes_gt)
        loss_bbox = loss_bbox * mask  # Zero out the loss for invalid boxes

        # Average the loss over the number of valid boxes
        loss_bbox = loss_bbox.sum() / mask.sum()

        # 3. Combine Losses
        loss = loss_seq + loss_bbox
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = image_features.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    epoch_loss = running_loss / total
    return epoch_loss

###############################################################################
#                        EVALUATION FUNCTION                                  #
###############################################################################
def evaluate(model, dataloader, criterion_seq, criterion_bbox, device):
    """
    Evaluation loop.
    Returns:
      - epoch_loss
      - average IoU
      - proportion of (correct program + correct answer),
        (correct program + incorrect answer),
        (incorrect program + correct answer),
        (incorrect program + incorrect answer)
      - program_token_accuracy (token-level accuracy on the program only)
    """
    model.eval()
    running_loss = 0.0
    total = 0

    # Accumulate IoU
    iou_cumulative = 0.0
    iou_batches    = 0

    # Track program/answer correctness
    cpca = 0  # correct program, correct answer
    cpia = 0  # correct program, incorrect answer
    ipca = 0  # incorrect program, correct answer
    ipia = 0  # incorrect program, incorrect answer

    # Program token-level accuracy
    program_token_correct = 0
    program_token_total   = 0  # (batch_size * 27)

    with torch.no_grad():
        for image_features, questions, combined_seq, bboxes_gt in tqdm(dataloader, desc="Evaluating", leave=False):
            batch_size = image_features.size(0)
            total += batch_size

            image_features = image_features.to(device)
            questions     = questions.to(device)
            combined_seq  = combined_seq.to(device)
            bboxes_gt     = bboxes_gt.to(device)

            seq_logits, bbox_preds = model(image_features, questions)

            # 1. Sequence Loss
            seq_logits_flat = seq_logits.reshape(-1, seq_logits.size(-1))
            targets_flat = combined_seq.reshape(-1)
            loss_seq = criterion_seq(seq_logits_flat, targets_flat)

            # 2. Bounding Box Loss (Smooth L1 Loss)
            mask = (bboxes_gt.sum(dim=2, keepdim=True) > 0)  # (batch,10,1)
            loss_bbox = criterion_bbox(bbox_preds, bboxes_gt)
            loss_bbox = loss_bbox * mask
            loss_bbox = loss_bbox.sum() / mask.sum()

            # 3. Combined Loss
            loss = loss_seq + loss_bbox
            running_loss += loss.item() * batch_size

            # 4. IoU Calculation
            mean_iou_val = batch_mean_iou(bbox_preds, bboxes_gt)
            iou_cumulative += mean_iou_val
            iou_batches += 1

            # 5. Program + Answer Correctness
            #    predicted_seq => (batch, 28)
            _, predicted_seq = torch.max(seq_logits, dim=2)  # (batch,28)

            # a) Program correctness
            program_pred = predicted_seq[:, :-1]  # shape (batch, 27)
            program_gt   = combined_seq[:, :-1]   # shape (batch, 27)
            # Exact match mask per sample
            program_exact = (program_pred == program_gt).all(dim=1)  # (batch,)

            # Token-level accuracy for program
            program_token_correct_batch = (program_pred == program_gt).sum().item()
            program_token_total_batch   = program_pred.numel()
            program_token_correct += program_token_correct_batch
            program_token_total   += program_token_total_batch

            # b) Answer correctness => last token
            answer_correct_mask  = (predicted_seq[:, -1] == combined_seq[:, -1])

            # Count correct/incorrect combinations
            for i in range(batch_size):
                prog_correct = bool(program_exact[i].item())
                ans_correct  = bool(answer_correct_mask[i].item())

                if prog_correct and ans_correct:
                    cpca += 1
                elif prog_correct and not ans_correct:
                    cpia += 1
                elif (not prog_correct) and ans_correct:
                    ipca += 1
                else:
                    ipia += 1

    # Final Statistics
    epoch_loss = running_loss / total
    avg_iou    = iou_cumulative / iou_batches if iou_batches > 0 else 0.0

    # Proportions
    prop_cpca = cpca / total
    prop_cpia = cpia / total
    prop_ipca = ipca / total
    prop_ipia = ipia / total

    # Program Token-Level Accuracy
    program_token_acc = program_token_correct / float(program_token_total) if program_token_total > 0 else 0.0

    return (epoch_loss, 
            avg_iou, 
            prop_cpca, 
            prop_cpia, 
            prop_ipca, 
            prop_ipia,
            program_token_acc)

###############################################################################
#                                 MAIN                                        #
###############################################################################
def main():
    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    # 1. Get Data Info
    vocab_size, program_vocab_size = get_data_info(Config.QUESTIONS_H5)
    Config.PROGRAM_VOCAB_SIZE = program_vocab_size
    print(f"Vocab Size: {vocab_size}, Program+Answer Vocab Size: {program_vocab_size}")

    # 2. Create Dataset Indices
    total_samples = 699989
    indices = list(range(total_samples))

    # 3. Train/Val/Test Splits
    train_val_indices, test_indices = train_test_split(
        indices, test_size=Config.TEST_SPLIT, random_state=Config.SEED
    )
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=Config.VALIDATION_SPLIT / (1 - Config.TEST_SPLIT),
        random_state=Config.SEED
    )

    print(f"Train samples: {len(train_indices)}, "
          f"Val samples: {len(val_indices)}, "
          f"Test samples: {len(test_indices)}")

    # 4. Create Datasets
    train_dataset = VQADataset(Config.FEATURES_H5, Config.QUESTIONS_H5, Config.BBOX_H5, train_indices)
    val_dataset   = VQADataset(Config.FEATURES_H5, Config.QUESTIONS_H5, Config.BBOX_H5, val_indices)
    test_dataset  = VQADataset(Config.FEATURES_H5, Config.QUESTIONS_H5, Config.BBOX_H5, test_indices)

    # 5. Create DataLoaders
    import multiprocessing
    num_workers = min(4, multiprocessing.cpu_count())
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 6. Device Configuration
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # 7. Initialize Model
    model = VQAModel(
        vocab_size=vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        program_vocab_size=Config.PROGRAM_VOCAB_SIZE,
        program_seq_len=Config.PROGRAM_SEQ_LEN,
        num_image_tokens=Config.NUM_IMAGE_TOKENS,
        special_token_id=Config.SPECIAL_TOKEN_ID
    ).to(device)

    # 8. Define Loss Functions and Optimizer
    criterion_seq = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss(reduction='none')  # For masked Smooth L1 loss
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 9. Training Parameters
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    patience = 5
    trigger_times = 0

    # 10. Training Loop
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")

        # Training
        train_loss = train_epoch(model, train_loader, criterion_seq, criterion_bbox, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation
        (val_loss, val_iou, 
         val_cpca, val_cpia, val_ipca, val_ipia,
         val_prog_token_acc) = evaluate(model, val_loader, criterion_seq, criterion_bbox, device)
        print(f"Val Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Program Token Acc: {val_prog_token_acc:.4f}")
        print(f"  CorrectProg+CorrectAns:   {val_cpca:.4f}")
        print(f"  CorrectProg+IncorrectAns: {val_cpia:.4f}")
        print(f"  IncorrectProg+CorrectAns: {val_ipca:.4f}")
        print(f"  IncorrectProg+IncorrectAns:{val_ipia:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            trigger_times = 0
            torch.save(model.state_dict(), Config.MODEL_NAME)
            print("Best model saved.")
        else:
            trigger_times += 1
            print(f"No improvement in val loss for {trigger_times} epoch(s).")
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

    # 11. Testing
    model.load_state_dict(best_model_state)
    (test_loss, test_iou, 
     test_cpca, test_cpia, test_ipca, test_ipia,
     test_prog_token_acc) = evaluate(model, test_loader, criterion_seq, criterion_bbox, device)
    print(f"\nTest Loss: {test_loss:.4f} | IoU: {test_iou:.4f} | Program Token Acc: {test_prog_token_acc:.4f}")
    print(f"  CorrectProg+CorrectAns:   {test_cpca:.4f}")
    print(f"  CorrectProg+IncorrectAns: {test_cpia:.4f}")
    print(f"  IncorrectProg+CorrectAns: {test_ipca:.4f}")
    print(f"  IncorrectProg+IncorrectAns:{test_ipia:.4f}")

if __name__ == "__main__":
    main()
