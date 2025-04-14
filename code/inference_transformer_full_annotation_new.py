import json
import h5py
import torch
import torch.nn as nn
import numpy as np
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# ---------------------------
# Positional Encoding Module
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ---------------------------
# Multi-modal Transformer Model
# ---------------------------
class MultiModalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=512, dropout=0.1, max_text_len=50, max_img_tokens=196):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.image_proj = nn.Linear(1024, d_model)
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_text_len+max_img_tokens)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len=max_text_len)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, dropout, batch_first=True)
        self.output_linear = nn.Linear(d_model, vocab_size)
    def forward(self, image_features, src_text, tgt_text):
        batch_size = image_features.size(0)
        img_feat = image_features.view(batch_size, 1024, 14*14).permute(0,2,1)  # (batch, 196, 1024)
        img_tokens = self.image_proj(img_feat)  # (batch, 196, d_model)
        src_emb = self.text_embedding(src_text)  # (batch, src_seq_len, d_model)
        encoder_input = torch.cat([img_tokens, src_emb], dim=1)  # (batch, 196+src_seq_len, d_model)
        encoder_input = self.pos_encoder(encoder_input)
        memory = self.transformer.encoder(encoder_input)
        tgt_emb = self.text_embedding(tgt_text)  # (batch, tgt_seq_len, d_model)
        tgt_emb = self.pos_decoder(tgt_emb)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(tgt_emb.device)
        output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        output = self.output_linear(output)
        return output

# ---------------------------
# Vocabulary Utilities
# ---------------------------
def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    rev_vocab = {int(v): k for k, v in vocab.items()}
    return vocab, rev_vocab

def decode_tokens(token_indices, rev_vocab):
    return " ".join([rev_vocab.get(idx, "<unk>") for idx in token_indices])

# ---------------------------
# Tokenization for Inference Chain
# ---------------------------
def tokenize_field(text: str, field: str) -> list:
    if field == "function":
        return [text] if text else []
    return re.findall(r'\[|\]|[^\[\]\s]+', text)

# ---------------------------
# Inference Chain Function
# ---------------------------
def run_inference_chain(model, image_features, final_chain, device, start_token, max_infer_len=20, rev_vocab=None):
    """
    model: the trained model.
    image_features: tensor of shape (1, 1024, 14, 14)
    final_chain: list of chain elements (each as a string of vocab indices) from final_chain_of_thought.
    device: computation device.
    start_token: token index to use as the start token.
    max_infer_len: maximum length for decoding each step.
    rev_vocab: reverse vocabulary mapping.
    Returns:
      final_output: predicted output (as a string of vocab indices) of the last chain step.
      cache: dict mapping chain step indices to predicted outputs.
    """
    model.eval()
    cache = {}
    for i, chain_elem in enumerate(final_chain):
        parts = chain_elem.strip().split()
        func_token = parts[0]
        input_step_indices = []
        for tok in parts[1:]:
            vocab_idx = int(tok)
            original_token = rev_vocab.get(vocab_idx, None)
            if original_token is not None and original_token.isdigit():
                input_step_indices.append(int(original_token))
            else:
                logging.warning(f"Token {tok} in chain step {i} is not recognized as a digit; skipping.")
        cached_inputs = []
        for idx in input_step_indices:
            if idx in cache:
                cached_inputs.append(cache[idx])
            else:
                logging.warning(f"Input index {idx} not found in cache for chain step {i}. Using empty string.")
                cached_inputs.append("")
        src_text_str = func_token + (" " + " ".join(cached_inputs) if cached_inputs else "")
        src_tokens = [int(tok) for tok in src_text_str.split()]
        src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
        pred_tokens = greedy_decode(model, image_features, src_tensor, start_token, max_infer_len, device)
        pred_tokens = pred_tokens.squeeze(0).tolist()
        cache[i] = " ".join(map(str, pred_tokens))
        logging.info(f"Chain step {i}: function token {func_token}, input steps {input_step_indices}, predicted output: {cache[i]}")
    final_output = cache[len(final_chain)-1]
    return final_output, cache

def greedy_decode(model, image_features, src_text, start_token, max_len, device):
    model.eval()
    with torch.no_grad():
        batch_size = src_text.size(0)
        img_feat = image_features.view(batch_size, 1024, 14*14).permute(0,2,1)
        img_tokens = model.image_proj(img_feat)
        src_emb = model.text_embedding(src_text)
        encoder_input = torch.cat([img_tokens, src_emb], dim=1)
        encoder_input = model.pos_encoder(encoder_input)
        memory = model.transformer.encoder(encoder_input)
        ys = torch.tensor([[start_token]], dtype=torch.long).to(device)
        for i in range(max_len - 1):
            tgt_emb = model.text_embedding(ys)
            tgt_emb = model.pos_decoder(tgt_emb)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
            out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            out = model.output_linear(out)
            prob = out[:, -1, :]
            next_word = torch.argmax(prob, dim=1).unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)
        return ys

# ---------------------------
# Main Inference Routine for 10 Examples
# ---------------------------
def main_inference():
    model_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/multimodal_transformer.pth"
    vocab_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/vocab.json"
    annotated_h5_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/annotated_questions_with_vocab.h5"
    features_h5_path = "/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5"

    d_model = 256
    nhead = 2
    num_encoder_layers = 1
    num_decoder_layers = 1
    dim_feedforward = 512
    dropout = 0.1
    max_src_len = 50
    max_infer_len = 20
    start_token = 0

    logging.info("Loading vocabulary...")
    vocab, rev_vocab = load_vocab(vocab_path)
    vocab_size = len(vocab)
    logging.info(f"Vocabulary loaded with {vocab_size} tokens.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info("Initializing model and loading checkpoint...")
    model = MultiModalTransformer(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                                  dim_feedforward, dropout, max_src_len, max_img_tokens=196).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logging.info("Model loaded.")

    logging.info("Loading annotated questions for inference...")
    with h5py.File(annotated_h5_path, 'r') as hf:
        questions_json = hf["questions"][()].decode("utf-8")
    annotated_data = json.loads(questions_json)["questions"]
    
    num_examples = 10
    logging.info(f"Performing inference on the first {num_examples} examples.")
    
    with h5py.File(features_h5_path, 'r') as hf:
        features = hf["features"]
        
        for i in range(min(num_examples, len(annotated_data))):
            question = annotated_data[i]
            logging.info(f"Processing question {i} with image_index {question['image_index']}.")
            actual_answer = question.get("answer", "Not provided")
            logging.info("Actual answer: " + actual_answer)
            final_chain = question["final_chain_of_thought"]
            # Load image features for this question's image.
            image_index = question["image_index"]
            img_feat = torch.tensor(features[image_index], dtype=torch.float).unsqueeze(0).to(device)
            
            final_output, cache = run_inference_chain(model, img_feat, final_chain, device, start_token, max_infer_len, rev_vocab)
            logging.info(f"Question {i} final predicted output (vocab indices): {final_output}")
            predicted_sentence = decode_tokens([int(tok) for tok in final_output.split()], rev_vocab)
            logging.info(f"Question {i} final predicted sentence: {predicted_sentence}")

if __name__ == "__main__":
    main_inference()
