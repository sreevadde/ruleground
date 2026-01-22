# RuleGround

**Bridging the Perceptual Gap: Rule-Grounded Representations for Tactical Reasoning in Multi-Sport Video Understanding**

Sree Krishna Vadde

---

## Overview

RuleGround is a modular neuro-symbolic architecture that addresses the **Perceptual Gap** in sports video understanding. Our analysis of 200 GPT-4o errors on the SportR benchmark shows that **62% are grounding failures** -- the model detects visual contact but misclassifies its rule-relevant status (e.g., contact observed but fails to determine whether the defender was in a legal guarding position).

RuleGround solves this by introducing an explicit **predicate bottleneck** between perception and reasoning, forcing the model to commit to structured intermediate representations that are then composed into rule evaluations via differentiable logic.

### Results on SportR-Hard

| Metric | Baseline (Qwen-VL-7B) | RuleGround | Delta |
|--------|----------------------|------------|-------|
| Q1 Infraction ID | 84.19% | **88.42 +/- 0.3%** | +4.23 |
| Q2 Foul Classification | 51.54% | **58.60 +/- 0.4%** | +7.06 |
| Q5 Temporal Grounding IoU | 9.94% | **14.12 +/- 0.2%** | +4.18 |
| False Positive Reduction | -- | **34%** | -- |

---

## Architecture

```
Video Clip
    |
    v
[Video Encoder]            Frozen VideoMAE-v2 ViT-B
    |
    v  Frame embeddings [B, T, D]
[Rule Grounding Module]
    |-- Temporal Attention Pool (RoPE + Flash Attention via ActionFormer v2)
    |-- Per-Predicate Classifiers (K=20 independent 2-layer MLPs)
    |-- SnapFormer Heads (frame-level instant predicate detection)
    |-- Sport-Conditional Masking
    |
    v  Predicate state p_hat in [0,1]^K
[Differentiable Logic]     Product t-norm: AND=a*b, OR=a+b-ab, NOT=1-a
    |
    v  Rule scores
[Reasoning Head]           Cross-attention transformer
    |
    v
Task Outputs: Q1 (infraction), Q2 (foul type), Q5 (temporal grounding)
```

### Training Pipeline (3 Stages)

1. **Supervised Pre-training**: `L = L_task + gamma * L_pred + delta * L_cons`
2. **GRPO Post-training**: Group-relative advantage estimation with predicate dropout for stochasticity
3. **RSA Risk Alignment**: CVaR penalty on false-positive infractions (34% FP reduction)

---

## Installation

```bash
# Clone
git clone https://github.com/sreevadde/ruleground.git
cd ruleground

# Install (requires Python >= 3.10)
pip install -e ".[dev]"

# Verify
python -c "from ruleground.models.ruleground import RuleGround; print('RuleGround installed')"
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.1.0
- [ActionFormer](https://github.com/sreevadde/actionformer) >= 1.5.0 (on PyPI)
- transformers, omegaconf, scikit-learn, typer

---

## Project Structure

```
ruleground/
├── models/                     # Model architectures
│   ├── ruleground.py           # Full model assembly
│   ├── encoder.py              # VideoMAE-v2 wrapper
│   ├── rgm.py                  # Rule Grounding Module
│   ├── temporal_pool.py        # Temporal attention pooling (ActionFormer v2)
│   ├── predicate_head.py       # Per-predicate classifiers
│   ├── snapformer_head.py      # Frame-level instant predicate detection
│   ├── logic.py                # Differentiable t-norm logic + rule composer
│   └── reasoning_head.py       # Multi-task cross-attention head (Q1/Q2/Q5)
│
├── predicates/                 # Predicate system
│   ├── ontology.py             # All 20 predicates (single source of truth)
│   ├── rules.py                # 12 rule definitions with formulas
│   └── extraction/             # LLM-based weak supervision
│       ├── extractor.py        # Claude/GPT-4o extraction
│       ├── prompts.py          # Prompt templates
│       └── validator.py        # Cross-model validation
│
├── data/                       # Data pipeline
│   ├── dataset.py              # SportR dataset loader
│   ├── transforms.py           # Video augmentations
│   └── collate.py              # Batch collation
│
├── training/                   # Training pipeline
│   ├── losses.py               # L_task + L_pred + L_cons + L_q5
│   ├── trainer.py              # Supervised trainer (Stage 1)
│   ├── grpo.py                 # GRPO with predicate dropout (Stage 2)
│   ├── rsa.py                  # RSA with CVaR penalty (Stage 3)
│   ├── rewards.py              # Multi-component reward function
│   └── callbacks.py            # W&B, early stopping, checkpointing
│
├── evaluation/                 # Evaluation
│   ├── metrics.py              # Q1/Q2/Q5/FP metrics
│   ├── evaluator.py            # Full evaluation pipeline
│   └── error_analysis.py       # Perception/grounding/reasoning attribution
│
├── utils/                      # Utilities
│   ├── config.py               # OmegaConf configuration
│   ├── logging.py              # Structured logging
│   ├── checkpoint.py           # Model checkpointing
│   └── distributed.py          # DDP utilities
│
└── cli/                        # Command-line interface
    ├── main.py                 # Entry point (typer)
    ├── train.py                # rg-train
    ├── eval.py                 # rg-eval
    └── extract.py              # rg-extract

configs/
├── base.yaml                   # Default configuration
├── model/{small,base,large}.yaml
├── training/{supervised,grpo,rsa}.yaml
└── sportr/{basketball,football,soccer}.yaml

tests/                          # 113 tests
├── models/                     # Logic, RGM, temporal pool, reasoning head
├── predicates/                 # Ontology verification
├── training/                   # Losses, GRPO, RSA, rewards
├── data/                       # Transforms, collation
├── test_metrics.py             # Q1/Q2/Q5 metrics
├── test_error_analysis.py      # Error attribution
└── test_config.py              # Configuration system
```

---

## Predicate Ontology (20 predicates)

| Sport | Predicate | Type |
|-------|-----------|------|
| Shared | `ball_in_play`, `contact_occurred`, `contact_before_arrival`, `incidental_contact` | State/Instant |
| Basketball | `defender_set`, `restricted_area`, `pivot_foot_lifted`, `ball_released`, `shooting_motion`, `verticality_maintained` | State/Instant/Spatial |
| Football | `ball_catchable`, `ball_in_air`, `offensive_push_off`, `within_five_yards` | State/Instant/Spatial |
| Soccer | `offside_position`, `involved_in_play`, `ball_contact_arm`, `arm_natural_position`, `played_by_opponent`, `denying_goal` | State/Instant/Spatial |

---

## Usage

### Training

```bash
# Stage 1: Supervised pre-training
rg-train -c configs/base.yaml -o experiments/run1

# Full 3-stage pipeline (supervised + GRPO + RSA)
ruleground train -c configs/base.yaml -o experiments/full --set training.use_grpo=true --set training.use_rsa=true
```

### Evaluation

```bash
ruleground eval -ckpt experiments/full/checkpoints/best.pt -c configs/base.yaml -s test
```

### Predicate Extraction

```bash
# Extract predicate labels from rationales using Claude
ruleground extract -d data/sportr/rationales.json -o data/sportr/predicates.json -b anthropic

# With cross-model validation
ruleground extract -d data/sportr/rationales.json -o data/sportr/predicates.json --validate
```

### Python API

```python
import torch
from ruleground.models.ruleground import RuleGround

model = RuleGround(
    encoder_name="MCG-NJU/videomae-base",
    freeze_encoder=True,
)

video = torch.randn(1, 16, 3, 224, 224)  # [B, T, C, H, W]
outputs = model(video)

# Task predictions
q1_pred = outputs["q1_logits"].argmax(dim=-1)   # infraction?
q2_pred = outputs["q2_logits"].argmax(dim=-1)   # which foul?
q5_span = outputs["q5_preds"]                    # temporal span

# Interpretable intermediate representations
predicates = outputs["predicate_probs"]           # [B, 20] predicate states
rules = outputs["rule_scores"]                    # {rule_name: [B] score}
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Specific module
pytest tests/models/ -v
pytest tests/training/ -v

# With coverage
pytest tests/ --cov=ruleground --cov-report=term-missing
```

---

## Citation

```bibtex
@article{vadde2026ruleground,
  title   = {Bridging the Perceptual Gap: Rule-Grounded Representations for
             Tactical Reasoning in Multi-Sport Video Understanding},
  author  = {Vadde, Sree Krishna},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## License

MIT
