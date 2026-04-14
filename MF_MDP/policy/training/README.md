# Policy Model Training

The Policy Actor model is designed to generate text responses while simultaneously predicting future collective opinion states using a dual-objective approach (text generation + state prediction).

## Architecture

```
Model Input:
├── input_ids: (B, seq_len)          # Tokenized input text
├── attention_mask: (B, seq_len)     # Attention mask
├── labels: (B, seq_len)             # Labels for text generation
├── current_state: (B, 3)            # Current opinion distribution [pos, neu, neg]
└── future_states: (B, K, 3)         # Future K-step state distributions

Model Architecture:
├── LoRA-finetuned LLM (Base Model)
│   ├── Target Modules: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
│   └── LoRA Config: rank=64, alpha=64
├── Hidden State Extraction
│   └── [EOS] token representation
└── Serial MLP Prediction Heads (×K)
    ├── Input: [llm_hidden + current_state + previous_preds]
    ├── Hidden: 512 units with ReLU + Dropout
    └── Output: 3 units (delta for state distribution)

Model Output:
├── text_loss: Cross-entropy with label smoothing
├── pred_loss: Weighted KL divergence over K steps
└── predicted_trajectories: (B, K, 3)  # Predicted state evolution
```

**Loss Function**: Dual-Objective with Soft Best-of-4 Sampling
```
L_text = Cross-entropy(logits, labels) with label smoothing
L_pred = E[∑_{k=1}^{K} γ^(k-1) * KL(s_k^* || ŝ_k)]
L_total = (1 - λ) * L_text + λ * L_pred
```

**Soft Best-of-4 Strategy**:
1. Generate 4 candidate features with different random noise seeds
2. Compute prediction loss for each candidate
3. Apply softmax weighting: α_i = softmax(-β * loss_i)
4. Combine losses: L_final = ∑ α_i * L_i (with detach on weights)

## Data Preparation

### Required Data Files

| Type | Format | Description | Example |
|------|--------|-------------|---------|
| CSV Data | `.csv` | User profile, topic, comments, and states | `policy_data.csv` |

### CSV Format

**Required columns**: `profile_text`, `topic`, `batch_mf`, `real_comments`, `pre_pos`, `pre_neg`, `pre_neu`

**Future states** (t0-t10): `dist_t0_pos`, `dist_t0_neg`, `dist_t0_neu`, ..., `dist_t10_pos`, `dist_t10_neg`, `dist_t10_neu`

```csv
profile_text,topic,batch_mf,real_comments,pre_pos,pre_neg,pre_neu,dist_t0_pos,dist_t0_neg,dist_t0_neu,...
```
*The policy is trained on actual state distributions from real data, avoiding the use of synthetic states from a learned model.*

## Training

### Basic Training

```bash
cd /root/ICML/release

python models/policy/training/train_policy.py \
    --pretrain /path/to/your/model \
    --dataset ./data/policy_data.csv \
    --output_dir ./checkpoints/policy \
    --epochs 3 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --k_steps 10
```

### Training with Soft Best-of-4

```bash
python models/policy/training/train_policy.py \
    --pretrain /path/to/your/model \
    --dataset ./data/policy_data.csv \
    --output_dir ./checkpoints/policy \
    --use_soft_best_of \
    --soft_best_beta 1.0 \
    --beta_schedule cosine \
    --beta_min 0.5 \
    --beta_max 5.0
```

### Resume Training

```bash
python models/policy/training/train_policy.py \
    --pretrain /path/to/your/model \
    --dataset ./data/policy_data.csv \
    --output_dir ./checkpoints/policy \
    --resume_from ./checkpoints/policy/last_checkpoint
```

## Configuration

### Command-Line Arguments

| Category | Argument | Description | Default |
|----------|----------|-------------|---------|
| **Model** | `--pretrain` | Pretrained model path (required) | - |
| | `--dataset` | Dataset CSV path (required) | - |
| | `--output_dir` | Output directory | `./checkpoints` |
| **LoRA** | `--lora_rank` | LoRA rank | 64 |
| | `--lora_alpha` | LoRA alpha | 64 |
| | `--target_modules` | Target modules | `all-linear` |
| **Training** | `--batch_size` | Batch size | 16 |
| | `--gradient_accumulation_steps` | Gradient accumulation | 4 |
| | `--learning_rate` | Learning rate | 1e-5 |
| | `--epochs` | Number of epochs | 3 |
| | `--max_len` | Max sequence length | 1024 |
| **Policy** | `--k_steps` | Future prediction steps | 10 |
| | `--lambda_coeff` | Loss balancing coefficient (λ) | 0.5 |
| | `--gamma` | Time decay coefficient | 0.9 |
| | `--label_smoothing` | Label smoothing | 0.1 |
| | `--loss_threshold` | KL loss threshold | 20.0 |
| **Soft Best-of-4** | `--use_soft_best_of` | Enable soft best-of-4 | False |
| | `--soft_best_beta` | Selection strength (β) | 1.0 |
| | `--beta_schedule` | Beta schedule | constant |
| | `--beta_min` | Minimum beta | 0.5 |
| | `--beta_max` | Maximum beta | 5.0 |
| **Other** | `--seed` | Random seed | 42 |
| | `--save_steps` | Save interval | 500 |
| | `--eval_steps` | Eval interval | 500 |
| | `--flash_attn` | Enable flash attention | False |
| | `--bf16` | Use BF16 | True |
| | `--use_wandb` | Enable WandB logging | False |
| | `--resume_from` | Resume checkpoint path | None |


## Beta Scheduling Strategies

The soft best-of-4 selection strength (β) can be scheduled during training:

| Strategy | Description |
|----------|-------------|
| `constant` | Fixed β value throughout training |
| `linear` | Linear increase from β_min to β_max |
| `cosine` | Cosine annealing from β_min to β_max |

Example:
```
β(t) = β_min + (β_max - β_min) * (1 - cos(π * t/T)) / 2
```
