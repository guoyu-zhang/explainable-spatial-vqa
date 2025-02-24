import os
import json
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

#############################################
# Global Function Vocabulary
#############################################
FUNCTION_VOCAB = {
    "END": 0,
    "scene": 1,
    "filter_size": 2,
    "filter_color": 3,
    "filter_material": 4,
    "filter_shape": 5,
    "relate": 6,
    "unique": 7,
    "count": 8,
    "query_color": 9,
    "query_shape": 10,
    "query_material": 11,
    "query_size": 12,
    "exist": 13
}
NUM_FUNCTIONS = len(FUNCTION_VOCAB)

#############################################
# Dataset: ClevrStepsDataset
#############################################
class ClevrStepsDataset(Dataset):
    def __init__(self, annotations_h5_path, image_features_h5_path):
        super().__init__()
        self.annotations_h5_path = annotations_h5_path
        self.image_features_h5_path = image_features_h5_path

        # Open the annotations HDF5 file and get the "questions" group.
        self.annotations_h5 = h5py.File(self.annotations_h5_path, "r")
        self.questions_group = self.annotations_h5["questions"]

        # Open the image features HDF5 file.
        self.img_feat_h5 = h5py.File(self.image_features_h5_path, "r")
        self.img_features_dataset = self.img_feat_h5["features"]  # shape: [70000, 1024, 14, 14]

        self.training_steps = []
        self._build_training_steps()

    def _build_training_steps(self):
        """
        Iterate over each question and each program step; for each step, save:
          - image_index, question_index, question_str, answer_str,
          - prog_tokens (chain-of-thought tokens),
          - input_values (the input bounding boxes for that step),
          - output_values (the output of the current step; e.g. a bounding box),
          - next_function (the function of the *next* step, or "END" if none).
        """
        for q_key in self.questions_group.keys():
            q_group = self.questions_group[q_key]
            image_index = int(q_group["image_index"][()])
            question_index = int(q_group["question_index"][()])
            question_str = q_group["question"][()].decode("utf-8")
            answer_str = q_group["answer"][()].decode("utf-8")
            annotated_program = q_group["annotated_program"]
            step_keys = sorted(annotated_program.keys())
            for i, sk in enumerate(step_keys):
                step_grp = annotated_program[sk]
                chain_str = step_grp["chain_of_thought"][()].decode("utf-8")
                prog_tokens = json.loads(chain_str)
                in_val_str = step_grp["input_values"][()].decode("utf-8")
                input_vals = json.loads(in_val_str)
                out_val_str = step_grp["output_values"][()].decode("utf-8")
                output_vals = json.loads(out_val_str)
                if i < len(step_keys) - 1:
                    next_step_key = step_keys[i+1]
                    next_step_grp = annotated_program[next_step_key]
                    if "function" in next_step_grp:
                        next_func = next_step_grp["function"][()].decode("utf-8")
                    else:
                        next_func = "END"
                else:
                    next_func = "END"
                sample = {
                    "image_index": image_index,
                    "question_index": question_index,
                    "question_str": question_str,
                    "answer_str": answer_str,
                    "prog_tokens": prog_tokens,
                    "input_values": input_vals,
                    "output_values": output_vals,
                    "next_function": next_func
                }
                self.training_steps.append(sample)
        print(f"Built {len(self.training_steps)} training steps.")

    def __len__(self):
        return len(self.training_steps)

    def __getitem__(self, idx):
        sample = self.training_steps[idx]
        img_idx = sample["image_index"]
        img_features = self.img_features_dataset[img_idx]
        img_features = torch.tensor(img_features, dtype=torch.float32)

        prog_tokens = sample["prog_tokens"]
        if prog_tokens and isinstance(prog_tokens[0], int):
            prog_tokens = torch.tensor(prog_tokens, dtype=torch.long)
        else:
            prog_tokens = torch.tensor([], dtype=torch.long)

        input_values = sample["input_values"]
        if isinstance(input_values, list) and len(input_values) > 0 and isinstance(input_values[0], list):
            input_values = torch.tensor(input_values, dtype=torch.float32)
        else:
            input_values = torch.tensor([], dtype=torch.float32)

        output_values = sample["output_values"]
        if isinstance(output_values, list) and len(output_values) == 4:
            output_values = torch.tensor(output_values, dtype=torch.float32)
        else:
            output_values = torch.tensor([], dtype=torch.float32)

        func_str = sample["next_function"]
        next_func_idx = FUNCTION_VOCAB.get(func_str, FUNCTION_VOCAB["END"])
        next_func_idx = torch.tensor(next_func_idx, dtype=torch.long)

        return {
            "img_features": img_features,    # [1024, 14, 14]
            "prog_tokens": prog_tokens,        # [L]
            "input_values": input_values,      # [N, 4]
            "output_values": output_values,    # [4]
            "question_str": sample["question_str"],
            "answer_str": sample["answer_str"],
            "question_index": sample["question_index"],
            "next_function": next_func_idx     # scalar label
        }

#############################################
# Model: CompositionalStepPredictor
#############################################
class CompositionalStepPredictor(nn.Module):
    def __init__(self, d_model=256, question_vocab_size=10000, prog_vocab_size=1000, num_functions=NUM_FUNCTIONS):
        """
        The model receives the following inputs:
          - Image features,
          - Question text,
          - Input values (bounding boxes), and
          - Chain-of-thought (prog tokens).
        It fuses these modalities and predicts:
          - Output values (e.g. a bounding box), and
          - The next function (a classification over function names).
        """
        super().__init__()
        self.d_model = d_model

        # Image encoder: average spatial features and project.
        self.image_fc = nn.Linear(1024, d_model)

        # Question encoder: using EmbeddingBag to average word embeddings.
        self.question_encoder = nn.Embedding(question_vocab_size, d_model)

        # For chain-of-thought tokens.
        self.prog_embedding = nn.Embedding(prog_vocab_size, d_model)

        # Input values (bounding boxes) encoder.
        self.input_encoder = nn.Linear(4, d_model)

        # Fusion layer: fuse image, question, input values, and chain-of-thought.
        self.fusion_fc = nn.Linear(d_model * 4, d_model)

        # Output heads.
        self.output_head = nn.Linear(d_model, 4)            # Predict output bounding box.
        self.function_head = nn.Linear(d_model, num_functions)  # Predict next function.
    
    def tokenize_question(self, question_str):
        """
        A simple tokenizer: splits on whitespace and maps each token to an index using a hash.
        In practice, you should use a proper vocabulary.
        """
        tokens = question_str.split()
        indices = [abs(hash(tok)) % 10000 for tok in tokens]
        return torch.tensor(indices, dtype=torch.long, device=self.start_token.device) if hasattr(self, 'start_token') else torch.tensor(indices, dtype=torch.long)

    def forward(self, img_features, question_str, input_values, prog_tokens):
        """
        Args:
            img_features: Tensor [1024, 14, 14].
            question_str: Python string.
            input_values: Tensor [N, 4] (can be empty).
            prog_tokens: Tensor [L] (chain-of-thought tokens; can be empty).
        Returns:
            pred_output: Tensor [4] (predicted output bounding box).
            pred_function_logits: Tensor [num_functions] (logits for next function).
        """
        # --- Encode Image Features ---
        f_img = img_features.mean(dim=(1, 2))  # Average spatially. [1024]
        f_img = self.image_fc(f_img)           # [d_model]

        # --- Encode Question ---
        q_tokens = question_str.split()
        q_indices = [abs(hash(tok)) % 10000 for tok in q_tokens]
        q_indices = torch.tensor(q_indices, dtype=torch.long, device=f_img.device)
        # When input is 2D, offsets should be None.
        f_question = self.question_encoder(q_indices.unsqueeze(0))
        f_question = f_question.squeeze(0)      # [d_model]

        # --- Encode Input Values (bounding boxes) ---
        if input_values.numel() > 0:
            f_input = self.input_encoder(input_values)  # [N, d_model]
            f_input = f_input.mean(dim=0)                # [d_model]
        else:
            f_input = torch.zeros(self.d_model, device=f_img.device)

        # --- Encode Chain-of-Thought ---
        if prog_tokens.numel() > 0:
            f_prog = self.prog_embedding(prog_tokens)      # [L, d_model]
            f_prog = f_prog.mean(dim=0)                     # [d_model]
        else:
            f_prog = torch.zeros(self.d_model, device=f_img.device)

        # --- Fuse Modalities ---
        fused = torch.cat([f_img, f_question, f_input, f_prog], dim=0)  # [4*d_model]
        fused = self.fusion_fc(fused)  # [d_model]

        # --- Output Predictions ---
        pred_output = self.output_head(fused)         # [4]
        pred_function_logits = self.function_head(fused)  # [num_functions]

        return pred_output, pred_function_logits

#############################################
# Training Loop with MPS Device Support
#############################################
def train_model():
    # Set device: use MPS if available, else fallback to CPU.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths to HDF5 files.
    annotations_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/output_full_annotations.h5"
    image_features_path = "/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5"

    # Create dataset and dataloader.
    dataset = ClevrStepsDataset(annotations_path, image_features_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    eval_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Instantiate model and move to device.
    model = CompositionalStepPredictor(d_model=256, question_vocab_size=10000, prog_vocab_size=1000, num_functions=NUM_FUNCTIONS)
    model.to(device)
    model.train()

    # Define optimizer and loss functions.
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    num_epochs = 2  # For demonstration.
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches = 0
        for step_idx, batch in enumerate(dataloader):
            # Move inputs to device.
            img_features = batch["img_features"].to(device)       # [1, 1024, 14, 14]
            question_str = batch["question_str"][0]                # string
            input_values = batch["input_values"].to(device)         # [N, 4] or empty
            prog_tokens = batch["prog_tokens"].to(device)           # [L] or empty
            gt_output = batch["output_values"].to(device)           # [4]
            gt_function = batch["next_function"].to(device)         # scalar

            optimizer.zero_grad()
            pred_output, pred_function_logits = model(img_features[0], question_str, input_values, prog_tokens)
            loss_output = mse_loss(pred_output, gt_output)
            loss_function = ce_loss(pred_function_logits.unsqueeze(0), gt_function.unsqueeze(0))
            loss = loss_output + loss_function
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if step_idx % 100 == 0 and step_idx > 0:
                print(f"Epoch {epoch}, Step {step_idx}, Loss={loss.item():.4f}")

        avg_loss = running_loss / num_batches if num_batches > 0 else 0
        test_prog_acc, cpca, cpia, ipca, ipia = evaluate_model(model, eval_dataloader, device, answer_threshold=0.01)
        print(f"Epoch {epoch} finished. Avg Loss={avg_loss:.4f}")
        print(f"ProgTokenAcc={test_prog_acc:.4f}")
        print(f"  CorrectProg+CorrectAns:   {cpca:.4f}")
        print(f"  CorrectProg+IncorrectAns: {cpia:.4f}")
        print(f"  IncorrectProg+CorrectAns: {ipca:.4f}")
        print(f"  IncorrectProg+IncorrectAns:{ipia:.4f}")

    print("Training completed!")
    model.eval()
    sample = next(iter(eval_dataloader))
    img_features_sample = sample["img_features"][0].to(device)
    question_sample = sample["question_str"][0]
    input_values_sample = sample["input_values"][0].to(device)
    prog_tokens_sample = sample["prog_tokens"][0].to(device) if sample["prog_tokens"].numel() > 0 else torch.tensor([], dtype=torch.long, device=device)
    pred_output, pred_function_logits = model(img_features_sample, question_sample, input_values_sample, prog_tokens_sample)
    pred_function = pred_function_logits.argmax(dim=-1).item()
    print("Generated output values (bbox):", pred_output.detach().cpu().tolist())
    print("Predicted next function (index):", pred_function)
    for func, idx in FUNCTION_VOCAB.items():
        if idx == pred_function:
            print("Predicted next function:", func)
            break

#############################################
# Evaluation Metrics
#############################################
def evaluate_model(model, dataloader, device, answer_threshold=0.01):
    """
    Evaluate the model:
      - ProgTokenAcc: computed as the fraction of samples where the predicted next function exactly matches ground truth.
      - Then, using MSE between predicted output (bbox) and ground truth output,
        classify the sample as a correct answer if error < threshold.
      - Compute:
          CPCA: Correct program + correct answer.
          CPIA: Correct program + incorrect answer.
          IPCA: Incorrect program + correct answer.
          IPIA: Incorrect program + incorrect answer.
    """
    model.eval()
    total_samples = 0
    correct_prog = 0
    count_cpca = 0
    count_cpia = 0
    count_ipca = 0
    count_ipia = 0
    mse = nn.MSELoss()

    with torch.no_grad():
        for batch in dataloader:
            img_features = batch["img_features"].to(device)
            question_str = batch["question_str"][0]
            input_values = batch["input_values"].to(device)
            prog_tokens = batch["prog_tokens"].to(device)
            gt_output = batch["output_values"].to(device)
            gt_function = batch["next_function"].to(device)
            pred_output, pred_function_logits = model(img_features[0], question_str, input_values, prog_tokens)
            pred_function = pred_function_logits.argmax(dim=-1).item()
            total_samples += 1
            if pred_function == gt_function.item():
                correct_prog += 1

            box_err = mse(pred_output, gt_output).item()
            answer_correct = (box_err < answer_threshold)
            prog_correct = (pred_function == gt_function.item())
            if prog_correct and answer_correct:
                count_cpca += 1
            elif prog_correct and (not answer_correct):
                count_cpia += 1
            elif (not prog_correct) and answer_correct:
                count_ipca += 1
            else:
                count_ipia += 1

    prog_acc = correct_prog / total_samples if total_samples > 0 else 0
    cpca = count_cpca / total_samples if total_samples > 0 else 0
    cpia = count_cpia / total_samples if total_samples > 0 else 0
    ipca = count_ipca / total_samples if total_samples > 0 else 0
    ipia = count_ipia / total_samples if total_samples > 0 else 0

    model.train()
    return prog_acc, cpca, cpia, ipca, ipia

#############################################
# Main
#############################################
if __name__ == "__main__":
    train_model()
