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
import re

# Import transformer modules
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

###############################################################################
#                              CONFIGURATION                                  #
###############################################################################
class Config:
    LAPTOP_OR_CLUSTER = 'L'
    PATH = '/exports/eddie/scratch/s1808795/vqa/code/' if LAPTOP_OR_CLUSTER == 'CLUSTER' else '/Users/guoyuzhang/University/Y5/diss/vqa/code/'

    FEATURES_H5 = (
        PATH + 'data/train_features.h5' 
        if LAPTOP_OR_CLUSTER == 'CLUSTER'
        else '/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5'
    )
    QUESTIONS_H5 = PATH + 'mapped_sequences.h5'
    MODELS_DIR = PATH + 'models'
    MODEL_NAME = PATH + 'models/cot_best_transformer_iqap.pth'

    BATCH_SIZE = 64
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 256
    IMAGE_FEATURE_DIM = 1024
    NUM_CLASSES = None
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    VALIDATION_SPLIT = 0.1
    TEST_SPLIT = 0.8
    SEED = 42

    PROGRAM_SEQ_LEN = 27
    PROGRAM_VOCAB_SIZE = None

    MAX_QUESTION_LEN = 46
    MAX_PROGRAM_LEN = 27
    NUM_IMAGE_TOKENS = 14 * 14  # 196
    SPECIAL_TOKEN_ID = 1  # <SOS> token ID

    COMBINED_SEQ_LEN = PROGRAM_SEQ_LEN + 1

    ALPHA_IOU = 1.0

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
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

###############################################################################
#                     HELPER: is_bbox_token, parse_bboxes, etc.              #
###############################################################################
def is_bbox_token(token_str):
    return bool(re.match(r'^[0-1]\.\d{3}$', token_str))

def parse_bboxes_from_seq(token_ids, id_to_token):
    tokens = [id_to_token.get(int(t.item()), "<UNK>") for t in token_ids]
    full_str = " ".join(tokens)
    pattern = r"\(\s*([0-1]\.\d{3})\s*,\s*([0-1]\.\d{3})\s*,\s*([0-1]\.\d{3})\s*,\s*([0-1]\.\d{3})\s*\)"
    bboxes = []
    for match in re.finditer(pattern, full_str):
        x1_str, y1_str, x2_str, y2_str = match.groups()
        x1 = float(x1_str); y1 = float(y1_str)
        x2 = float(x2_str); y2 = float(y2_str)
        bboxes.append((x1, y1, x2, y2))
    return bboxes

def iou(boxA, boxB):
    (Axmin, Aymin, Axmax, Aymax) = boxA
    (Bxmin, Bymin, Bxmax, Bymax) = boxB
    inter_xmin = max(Axmin, Bxmin)
    inter_ymin = max(Aymin, Bymin)
    inter_xmax = min(Axmax, Bxmax)
    inter_ymax = min(Aymax, Bymax)
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    areaA = max(0.0, Axmax - Axmin) * max(0.0, Aymax - Aymin)
    areaB = max(0.0, Bxmax - Bxmin) * max(0.0, Bymax - Bymin)
    union_area = areaA + areaB - inter_area
    if union_area < 1e-9:
        return 0.0
    return inter_area / union_area

def cross_entropy_ignore_bbox(logits, targets, id_to_token):
    """
    If token is a bounding-box coordinate e.g. '0.123', we skip it for CE.
    """
    device = logits.device
    B = targets.size(0)
    keep_indices = []
    for i in range(B):
        t_id = int(targets[i].item())
        token_str = id_to_token.get(t_id, "<UNK>")
        if not is_bbox_token(token_str):
            keep_indices.append(i)
    if len(keep_indices) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    keep_indices_t = torch.tensor(keep_indices, dtype=torch.long, device=device)
    logits_sub = torch.index_select(logits, 0, keep_indices_t)
    targets_sub= torch.index_select(targets, 0, keep_indices_t)
    return nn.functional.cross_entropy(logits_sub, targets_sub)

###############################################################################
#                               DATASET                                       #
###############################################################################
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

        # 1) Image features
        image_idx = self.questions_file['image_index'][actual_idx]
        image_features = self.features_file['features'][image_idx]
        image_features = torch.tensor(image_features, dtype=torch.float32)
        image_features = image_features.permute(1, 2, 0).contiguous().view(-1, 1024)

        # 2) question
        question = self.questions_file['question_tokens'][actual_idx]
        question = torch.tensor(question, dtype=torch.long)

        # 3) program (27 tokens)
        program = self.questions_file['program_tokens'][actual_idx]
        program = torch.tensor(program, dtype=torch.long)

        # 4) answer (1 token), or maybe multiple but we take first
        answer = self.questions_file['answer_tokens'][actual_idx]
        answer = torch.tensor(answer, dtype=torch.long)
        answer = answer[0].unsqueeze(0)  # (1,)

        # combine => shape(28,)
        combined_seq = torch.cat([program, answer], dim=0)

        return image_features, question, combined_seq

    def __del__(self):
        if self.features_file is not None:
            self.features_file.close()
        if self.questions_file is not None:
            self.questions_file.close()

###############################################################################
#                          MODEL DEFINITION                                   #
###############################################################################
class VQAModel(nn.Module):
    """
    Transformer that outputs a program+answer of length (program_seq_len+1).
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
        self.seq_len = program_seq_len + 1

        # 1) Image projection
        self.image_proj = nn.Linear(Config.IMAGE_FEATURE_DIM, embedding_dim)
        # 2) question embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 3) a special CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # 4) PosEnc for encoder
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=0.1,
                                             max_len=num_image_tokens + Config.MAX_QUESTION_LEN + 1)
        # 5) TransformerEncoder
        encoder_layers = TransformerEncoderLayer(d_model=embedding_dim, nhead=4)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=1)

        # 6) TransformerDecoder for combined sequence
        self.decoder_embedding = nn.Embedding(program_vocab_size, embedding_dim, padding_idx=0)
        self.pos_decoder = PositionalEncoding(embedding_dim, dropout=0.1, max_len=self.seq_len+1)
        decoder_layers = TransformerDecoderLayer(d_model=embedding_dim, nhead=4)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers=1)
        self.output_layer = nn.Linear(embedding_dim, program_vocab_size)

    def forward(self, image_features, questions):
        batch_size = image_features.size(0)

        image_encoded = self.image_proj(image_features)
        question_embedded = self.embedding(questions)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        encoder_input = torch.cat((cls_tokens, image_encoded, question_embedded), dim=1)
        encoder_input = encoder_input.permute(1,0,2)
        encoder_input = self.pos_encoder(encoder_input)
        memory = self.transformer_encoder(encoder_input)

        seq_logits = self.autoregressive_decode(memory, self.seq_len)
        return seq_logits

    def autoregressive_decode(self, memory, seq_len):
        batch_size = memory.size(1)
        device = memory.device

        sos_tokens = torch.full((batch_size,1), Config.SPECIAL_TOKEN_ID,
                                dtype=torch.long, device=device)
        generated_seq = sos_tokens
        all_logits= []

        for _ in range(seq_len):
            embedded_seq = self.decoder_embedding(generated_seq)
            embedded_seq = embedded_seq.permute(1,0,2)
            embedded_seq = self.pos_decoder(embedded_seq)

            tgt_seq_len = embedded_seq.size(0)
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)

            dec_out= self.transformer_decoder(
                tgt=embedded_seq,
                memory=memory,
                tgt_mask=tgt_mask
            )
            output_logits= self.output_layer(dec_out[-1,:,:])
            all_logits.append(output_logits.unsqueeze(1))
            _, next_tokens= torch.max(output_logits, dim=1)
            next_tokens= next_tokens.unsqueeze(1)
            generated_seq= torch.cat([generated_seq, next_tokens], dim=1)

        seq_logits= torch.cat(all_logits, dim=1)
        return seq_logits

###############################################################################
#                            UTILITY: get_data_info                           #
###############################################################################
def get_data_info(questions_h5_path):
    import numpy as np
    with h5py.File(questions_h5_path, 'r') as f:
        questions= f['question_tokens']
        answers  = f['answer_tokens']
        programs = f['program_tokens']
        vocab_size= int(np.max(questions))+1
        program_vocab_size= max(int(np.max(programs)), int(np.max(answers)))+1
    return vocab_size, program_vocab_size

###############################################################################
#                       DECODING HELPERS & ACCURACY                           #
###############################################################################
def measure_program_accuracy_ignoring_bboxes_and_answer(pred_seq, gt_seq, id_to_token):
    ans_ok= (pred_seq[-1].item() == gt_seq[-1].item())
    pred_prog= pred_seq[:-1]
    gt_prog= gt_seq[:-1]
    if len(pred_prog)!= len(gt_prog):
        return (False, ans_ok)
    for i in range(len(gt_prog)):
        gt_id= int(gt_prog[i].item())
        # skip if it's bounding-box token
        if is_bbox_token(id_to_token.get(gt_id,"<UNK>")):
            continue
        if pred_prog[i].item()!= gt_id:
            return (False, ans_ok)
    return (True, ans_ok)


### NEW: let's define a decode function for printing
def decode_ids(ids, id_to_token):
    """ Convert a 1D tensor of token IDs into text tokens. """
    return [id_to_token.get(int(x.item()), "<UNK>") for x in ids]


###############################################################################
#                    TRAIN / EVAL (with printing one sample)                  #
###############################################################################
def train_epoch(model, dataloader, criterion_seq, optimizer, device, id_to_token, epoch):
    model.train()
    running_ce_loss=0.0
    running_bbox_loss=0.0
    running_total_loss=0.0
    total_samples=0

    sum_iou=0.0
    iou_count=0

    correct_prog_and_correct_answer=0
    correct_prog_and_incorrect_answer=0
    incorrect_prog_and_correct_answer=0
    incorrect_prog_and_incorrect_answer=0

    correct_prog_tokens=0
    total_prog_tokens=0

    ### NEW: We'll store one sample from the last batch so we can decode & print
    example_question= None
    example_gt_seq= None
    example_pred_seq= None

    for batch_idx, (image_features, questions, combined_seq) in enumerate(tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)):
        image_features= image_features.to(device)
        questions= questions.to(device)
        combined_seq= combined_seq.to(device)

        bsz= image_features.size(0)
        total_samples+= bsz

        optimizer.zero_grad()
        seq_logits= model(image_features, questions)  # (bsz, 28, vocab_size)

        # Flatten ignoring bounding boxes
        seq_logits_flat= seq_logits.view(-1, seq_logits.size(-1))
        targets_flat= combined_seq.view(-1)
        ce_loss= cross_entropy_ignore_bbox(seq_logits_flat, targets_flat, id_to_token)

        _, pred_seq= torch.max(seq_logits, dim=2)

        # IoU bounding box calc
        bbox_sum= 0.0
        bbox_count=0
        with torch.no_grad():
            for i in range(bsz):
                pred_boxes= parse_bboxes_from_seq(pred_seq[i], id_to_token)
                gt_boxes= parse_bboxes_from_seq(combined_seq[i], id_to_token)
                if len(pred_boxes)>0 and len(gt_boxes)>0:
                    pairs= min(len(pred_boxes), len(gt_boxes))
                    ious= []
                    for j in range(pairs):
                        ious.append(iou(pred_boxes[j], gt_boxes[j]))
                    mean_iou= sum(ious)/ len(ious)
                    bbox_sum+= (1.0- mean_iou)
                    bbox_count+=1

        if bbox_count>0:
            avg_bbox_loss= bbox_sum/ bbox_count
        else:
            avg_bbox_loss= 0.0
        bbox_loss= torch.tensor(avg_bbox_loss, dtype=torch.float32, device=device, requires_grad=True)
        total_loss= ce_loss + Config.ALPHA_IOU*bbox_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_ce_loss   += ce_loss.item()* bsz
        running_bbox_loss += avg_bbox_loss  * bsz
        running_total_loss+= total_loss.item()* bsz

        # measure correctness
        for i in range(bsz):
            prog_ok, ans_ok= measure_program_accuracy_ignoring_bboxes_and_answer(
                pred_seq[i], combined_seq[i], id_to_token
            )
            if   prog_ok and ans_ok:
                correct_prog_and_correct_answer+=1
            elif prog_ok and not ans_ok:
                correct_prog_and_incorrect_answer+=1
            elif (not prog_ok) and ans_ok:
                incorrect_prog_and_correct_answer+=1
            else:
                incorrect_prog_and_incorrect_answer+=1

            # program token-level ignoring bboxes
            pseq= pred_seq[i][:-1]
            gseq= combined_seq[i][:-1]
            L= min(len(pseq), len(gseq))
            for k in range(L):
                gt_id= int(gseq[k].item())
                if is_bbox_token(id_to_token.get(gt_id,"<UNK>")):
                    continue
                if pseq[k].item()== gt_id:
                    correct_prog_tokens+=1
                total_prog_tokens+=1

        ### NEW: store example from the last batch in the loop
        # We'll store the first sample from this batch
        example_question= questions[0].detach().cpu()
        example_gt_seq= combined_seq[0].detach().cpu()
        example_pred_seq= pred_seq[0].detach().cpu()

    # after all batches
    epoch_ce_loss= running_ce_loss/ total_samples
    epoch_bbox_loss= running_bbox_loss/ total_samples
    epoch_total_loss= running_total_loss/ total_samples

    avg_iou= sum_iou/ iou_count if iou_count>0 else 0.0
    prog_token_acc= (correct_prog_tokens/ total_prog_tokens) if total_prog_tokens>0 else 0.0

    total_batches= total_samples
    p_cpca= correct_prog_and_correct_answer/ total_batches
    p_cpia= correct_prog_and_incorrect_answer/ total_batches
    p_ipca= incorrect_prog_and_correct_answer/ total_batches
    p_ipia= incorrect_prog_and_incorrect_answer/ total_batches

    print(f"Train => CE Loss={epoch_ce_loss:.4f}, BBox Loss={epoch_bbox_loss:.4f}, "
          f"Total Loss={epoch_total_loss:.4f}, IoU={avg_iou:.4f}, Program Token Acc={prog_token_acc:.4f}")
    print(f"  CorrectProg+CorrectAns:   {p_cpca:.4f}")
    print(f"  CorrectProg+IncorrectAns: {p_cpia:.4f}")
    print(f"  IncorrectProg+CorrectAns: {p_ipca:.4f}")
    print(f"  IncorrectProg+IncorrectAns:{p_ipia:.4f}")

    ### NEW: decode & print one sample from this epoch
    if example_question is not None:
        q_str   = decode_ids(example_question, id_to_token)
        gt_str  = decode_ids(example_gt_seq, id_to_token)
        pred_str= decode_ids(example_pred_seq, id_to_token)
        print("\n### SAMPLE FROM EPOCH", epoch, "###")
        print("Q  :", " ".join(q_str))
        print("GT :", " ".join(gt_str))
        print("PR :", " ".join(pred_str))
        print("###########################\n")

    return (epoch_ce_loss, epoch_bbox_loss, epoch_total_loss,
            avg_iou, prog_token_acc,
            p_cpca, p_cpia, p_ipca, p_ipia)

def evaluate(model, dataloader, criterion_seq, device,
             id_to_token, epoch):
    model.eval()
    running_ce_loss= 0.0
    running_bbox_loss=0.0
    running_total_loss=0.0
    total_samples=0

    sum_iou=0.0
    iou_count=0

    correct_prog_and_correct_answer=0
    correct_prog_and_incorrect_answer=0
    incorrect_prog_and_correct_answer=0
    incorrect_prog_and_incorrect_answer=0

    correct_prog_tokens=0
    total_prog_tokens=0

    ### NEW: also store example from validation
    example_question= None
    example_gt_seq= None
    example_pred_seq= None

    with torch.no_grad():
        for batch_idx, (image_features, questions, combined_seq) in enumerate(tqdm(dataloader, desc=f"Eval Epoch {epoch}", leave=False)):
            image_features= image_features.to(device)
            questions= questions.to(device)
            combined_seq= combined_seq.to(device)

            bsz= image_features.size(0)
            total_samples+= bsz

            seq_logits= model(image_features, questions)
            seq_logits_flat= seq_logits.view(-1, seq_logits.size(-1))
            targets_flat= combined_seq.view(-1)
            ce_loss= cross_entropy_ignore_bbox(seq_logits_flat, targets_flat, id_to_token)

            _, predicted_seq= torch.max(seq_logits, dim=2)
            bbox_sum=0.0
            bbox_count=0
            for i in range(bsz):
                pred_boxes= parse_bboxes_from_seq(predicted_seq[i], id_to_token)
                gt_boxes  = parse_bboxes_from_seq(combined_seq[i], id_to_token)
                if len(pred_boxes)>0 and len(gt_boxes)>0:
                    pairs= min(len(pred_boxes), len(gt_boxes))
                    ious=[]
                    for j in range(pairs):
                        ious.append(iou(pred_boxes[j], gt_boxes[j]))
                    mean_iou= sum(ious)/len(ious)
                    bbox_sum+= (1.0-mean_iou)
                    bbox_count+=1

                # measure correctness
                prog_ok, ans_ok= measure_program_accuracy_ignoring_bboxes_and_answer(
                    predicted_seq[i], combined_seq[i], id_to_token
                )
                if   prog_ok and ans_ok:
                    correct_prog_and_correct_answer+=1
                elif prog_ok and not ans_ok:
                    correct_prog_and_incorrect_answer+=1
                elif (not prog_ok) and ans_ok:
                    incorrect_prog_and_correct_answer+=1
                else:
                    incorrect_prog_and_incorrect_answer+=1

                # program token-level ignoring bboxes & last
                pseq= predicted_seq[i][:-1]
                gseq= combined_seq[i][:-1]
                L= min(len(pseq), len(gseq))
                for k in range(L):
                    gt_id= int(gseq[k].item())
                    if is_bbox_token(id_to_token.get(gt_id,"<UNK>")):
                        continue
                    if pseq[k].item()== gt_id:
                        correct_prog_tokens+=1
                    total_prog_tokens+=1

            if bbox_count>0:
                avg_bbox= bbox_sum/ bbox_count
            else:
                avg_bbox=0.0
            total_loss= ce_loss + Config.ALPHA_IOU* avg_bbox

            running_ce_loss   += ce_loss.item()* bsz
            running_bbox_loss += avg_bbox     * bsz
            running_total_loss+= total_loss.item()* bsz

            ### NEW: store first sample from last batch
            example_question= questions[0].detach().cpu()
            example_gt_seq= combined_seq[0].detach().cpu()
            example_pred_seq= predicted_seq[0].detach().cpu()

    epoch_ce_loss= running_ce_loss / total_samples
    epoch_bbox_loss= running_bbox_loss / total_samples
    epoch_total_loss= running_total_loss / total_samples

    avg_iou= sum_iou/ iou_count if iou_count>0 else 0.0
    prog_token_acc= (correct_prog_tokens/ total_prog_tokens) if total_prog_tokens>0 else 0.0

    cpca= correct_prog_and_correct_answer
    cpia= correct_prog_and_incorrect_answer
    ipca= incorrect_prog_and_correct_answer
    ipia= incorrect_prog_and_incorrect_answer
    total_batches= total_samples
    p_cpca= cpca / total_batches
    p_cpia= cpia / total_batches
    p_ipca= ipca / total_batches
    p_ipia= ipia / total_batches

    print(f"Eval => CE Loss: {epoch_ce_loss:.4f} | IoU: {avg_iou:.4f} | Program Token Acc: {prog_token_acc:.4f}")
    print(f"  CorrectProg+CorrectAns:   {p_cpca:.4f}")
    print(f"  CorrectProg+IncorrectAns: {p_cpia:.4f}")
    print(f"  IncorrectProg+CorrectAns: {p_ipca:.4f}")
    print(f"  IncorrectProg+IncorrectAns:{p_ipia:.4f}")

    ### NEW: print example
    if example_question is not None:
        q_str   = decode_ids(example_question, id_to_token)
        gt_str  = decode_ids(example_gt_seq, id_to_token)
        pred_str= decode_ids(example_pred_seq, id_to_token)
        print("\n### VAL SAMPLE FROM EPOCH", epoch, "###")
        print("Q  :", " ".join(q_str))
        print("GT :", " ".join(gt_str))
        print("PR :", " ".join(pred_str))
        print("###########################\n")

    return (epoch_ce_loss, epoch_bbox_loss, epoch_total_loss,
            avg_iou, prog_token_acc,
            p_cpca, p_cpia, p_ipca, p_ipia)

###############################################################################
#                                  MAIN                                       #
###############################################################################
def main():
    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    vocab_size, program_vocab_size= get_data_info(Config.QUESTIONS_H5)
    Config.PROGRAM_VOCAB_SIZE= program_vocab_size
    print(f"Vocab Size: {vocab_size}, Combined Program+Answer Vocab Size: {program_vocab_size}")

    total_samples= 699989
    indices= list(range(total_samples))

    from sklearn.model_selection import train_test_split
    train_val_indices, test_indices= train_test_split(indices, test_size=Config.TEST_SPLIT, random_state=Config.SEED)
    train_indices, val_indices= train_test_split(
        train_val_indices,
        test_size= Config.VALIDATION_SPLIT / (1 - Config.TEST_SPLIT),
        random_state= Config.SEED
    )
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}, Test samples: {len(test_indices)}")

    train_dataset= VQADataset(Config.FEATURES_H5, Config.QUESTIONS_H5, train_indices)
    val_dataset  = VQADataset(Config.FEATURES_H5, Config.QUESTIONS_H5, val_indices)
    test_dataset = VQADataset(Config.FEATURES_H5, Config.QUESTIONS_H5, test_indices)

    import multiprocessing
    num_workers= min(4, multiprocessing.cpu_count())
    train_loader= DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    val_loader  = DataLoader(val_dataset,   batch_size=Config.BATCH_SIZE, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset,  batch_size=Config.BATCH_SIZE, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    device= torch.device('mps' if torch.backends.mps.is_available()
                         else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print("Using device:", device)

    model= VQAModel(
        vocab_size= vocab_size,
        embedding_dim= Config.EMBEDDING_DIM,
        hidden_dim= Config.HIDDEN_DIM,
        program_vocab_size= Config.PROGRAM_VOCAB_SIZE,
        program_seq_len= Config.PROGRAM_SEQ_LEN,
        num_image_tokens= Config.NUM_IMAGE_TOKENS,
        special_token_id= Config.SPECIAL_TOKEN_ID
    ).to(device)

    id_to_token= {i: f"TOK_{i}" for i in range(Config.PROGRAM_VOCAB_SIZE)}

    criterion_seq= nn.CrossEntropyLoss(reduction='mean')
    optimizer= torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_seq_loss= float('inf')
    best_model_state= copy.deepcopy(model.state_dict())
    patience= 10
    trigger_times= 0

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")

        # Train
        (train_ce_loss, train_bbox_loss, train_total_loss,
         train_iou, train_prog_acc,
         train_cpca, train_cpia, train_ipca, train_ipia) = train_epoch(
            model, train_loader, criterion_seq, optimizer, device, id_to_token, epoch
        )

        # Validate
        (val_ce_loss, val_bbox_loss, val_total_loss,
         val_iou, val_prog_acc,
         val_cpca, val_cpia, val_ipca, val_ipia) = evaluate(
            model, val_loader, criterion_seq, device, id_to_token, epoch
        )

        print(f"Val => CE Loss={val_ce_loss:.4f}, IoU={val_iou:.4f}, Program Token Acc={val_prog_acc:.4f}")
        print(f"  CorrectProg+CorrectAns:   {val_cpca:.4f}")
        print(f"  CorrectProg+IncorrectAns: {val_cpia:.4f}")
        print(f"  IncorrectProg+CorrectAns: {val_ipca:.4f}")
        print(f"  IncorrectProg+IncorrectAns:{val_ipia:.4f}")

        # Early stopping
        if val_ce_loss < best_val_seq_loss:
            best_val_seq_loss= val_ce_loss
            best_model_state= copy.deepcopy(model.state_dict())
            trigger_times= 0
            torch.save(model.state_dict(), Config.MODEL_NAME)
            print("Best model saved.")
        else:
            trigger_times+=1
            print(f"No improvement in val seq CE for {trigger_times} epoch(s).")
            if trigger_times>= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

    # test
    model.load_state_dict(best_model_state)
    (test_ce_loss, test_bbox_loss, test_total_loss,
     test_iou, test_prog_acc,
     test_cpca, test_cpia, test_ipca, test_ipia)= evaluate(
        model, test_loader, criterion_seq, device, id_to_token, epoch="TEST"
    )
    print(f"\nTest => CE={test_ce_loss:.4f}, BBox={test_bbox_loss:.4f}, Total={test_total_loss:.4f}, IoU={test_iou:.4f}, "
          f"ProgTokenAcc={test_prog_acc:.4f}")
    print(f"  CorrectProg+CorrectAns:   {test_cpca:.4f}")
    print(f"  CorrectProg+IncorrectAns: {test_cpia:.4f}")
    print(f"  IncorrectProg+CorrectAns: {test_ipca:.4f}")
    print(f"  IncorrectProg+IncorrectAns:{test_ipia:.4f}")

if __name__== "__main__":
    main()
