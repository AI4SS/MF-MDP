# State Transition Model Training

The State Transition Network (ST-Net) is designed to model the temporal evolution of collective opinion state distributions within a population.

## Architecture

```
Model Input:
├── mu_prev_seq: (B, T, 3)        # Historical state distribution
├── text_emb_seq: (B, T, 768)     # Mean field text embeddings
└── agent_feat_seq: (B, T, 768)   # Agent features (pooled)

Model Architecture:
├── Input Embedding Projection (768 -> 256)
├── Positional Encoding
├── Causal Transformer Layers (×N)
│   ├── Multi-Head Self-Attention
│   ├── Feed-Forward Network
│   └── Layer Normalization
└── Output Head (256 -> 3)

Model Output:
└── mu_pred_seq: (B, T, 3)        # Predicted state distribution
```

**Loss Function**: KL Divergence
```
L = E[∑_{t=1}^{T} KL(m_t^* || m̂_t)]
```

## Data Preparation

### Required Data Files

| Type | Format | Description | Example |
|------|--------|-------------|---------|
| Event Data | `.json` | User interaction stream | `jiangping.json` |
| Mean Field | `*_mf.csv` | Opinion summary text | `jiangping_mf.csv` |
| State Trajectory | `*_trajectory.csv` | State distribution evolution | `jiangping_trajectory.csv` |

### State Trajectory  Format

```csv
batch_id,batch_ratio_pos,batch_ratio_neu,batch_ratio_neg,cum_ratio_pos,cum_ratio_neu,cum_ratio_neg
0,0.25,0.45,0.30,0.25,0.45,0.30
1,0.23,0.47,0.30,0.24,0.46,0.30
...
```

## Training

### Basic Training

```bash
cd /root/ICML/release

python models/state_transition/training/train_event_transformer.py \
    --epochs 20 \
    --batch_size 4 \
    --max_event 100
```

### Resume Training

```bash
python models/state_transition/training/train_event_transformer.py \
    --epochs 20 \
    --batch_size 4 \
    --resume \
    --checkpoint checkpoints/event/checkpoint_last.pt
```

## Configuration

Edit `TrainConfig` in `train_event_transformer.py`:

```python
@dataclass
class TrainConfig:
    # Data paths
    event_data_dir: str = "/path/to/event/json/files"
    mf_dir: str = "/path/to/mf/csv/files"
    state_trajectory_dir: str = "/path/to/trajectory/csv/files"

    # Model parameters
    d_model: int = 256          # Transformer hidden size
    nhead: int = 8              # Number of attention heads
    num_layers: int = 3         # Number of transformer layers
    text_emb_dim: int = 768     # BERT embedding dimension
    agent_feat_dim: int = 768   # Agent profile feature dimension

    # Training parameters
    train_batch_size: int = 4   # Event-level batch size
    num_agents: int = 16        # Agents per batch
    lr: float = 2e-5            # Learning rate
    num_epochs: int = 20
    max_event: int = 100        # Max events to use (excluding English events)

    # Save paths
    save_dir: str = "./checkpoints/event"
    save_name: str = "event_transformer_best.pt"
```

## Training Output

| Output | Description |
|--------|-------------|
| `checkpoints/event/event_transformer_best.pt` | Best model |
| `checkpoints/event/checkpoint_last.pt` | Latest checkpoint |
| `checkpoints/event/runs/` | TensorBoard logs |
| `cache/text_embeddings/` | Pre-encoded text embeddings |

### View Logs

```bash
tensorboard --logdir checkpoints/event/runs
```

## Generate Predictions

After training, generate state distribution predictions:

```bash
python main/script/generate_state_trajectory.py \
    --checkpoint checkpoints/event/event_transformer_best.pt \
    --mf_dir ./data/test_mf \
    --traj_dir ./data/test_state_distribution \
    --output_dir ./data/pred_state_distribution \
    --cache_dir ./cache/text_embeddings \
    --batch_size 16 \
    --warmup_steps 5 \
    --device cuda
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--checkpoint` | Model checkpoint path (required) | - |
| `--event_name` | Single event name, or process all | None |
| `--warmup_steps` | Warmup steps using ground truth | 5 |
| `--mf_dir` | Mean field CSV directory | `./data/test_mf` |
| `--traj_dir` | Trajectory CSV directory | `./data/test_state_distribution` |
| `--output_dir` | Output directory | `./data/pred_state_distribution` |
| `--cache_dir` | Text embedding cache directory | `./cache/text_embeddings` |

## Notes

1. **English Events**: Events starting with English letters are always included (no `max_event` limit)
2. **Pre-encoding**: MF files are pre-encoded and cached for efficiency
3. **GPU Memory**: Training requires 8GB+ GPU memory
4. **Logging**: Uses both wandb and TensorBoard for experiment tracking
