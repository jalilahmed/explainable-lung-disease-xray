# Baseline Evaluation of ResNet-50 on NIH ChestX-ray14

This repository contains a **baseline multilabel classification experiment** using a standard ResNet-50 trained on the NIH ChestX-ray14 dataset.  
The goal of this project is not to achieve state-of-the-art performance, but to demonstrate **correct problem setup, evaluation practices, and cautious use of interpretability tools** on a widely used medical imaging benchmark.

**Website:** https://jalilahmed.github.io 
**Blog post and Report:** https://jalilahmed.github.io/notes/note2/

---

## Motivation

Chest X-ray datasets are frequently used to showcase model performance and interpretability claims. However, many issues—such as data leakage, label noise, and overinterpretation of explanations—are often overlooked.

This project aims to:
- implement a clean, reproducible baseline,
- evaluate performance appropriately for a multilabel medical task,
- and illustrate what can and cannot be inferred from common interpretability methods.

---

## Dataset

- **NIH ChestX-ray14**
- All images and labels are used as provided.
- **Official patient-wise train/validation/test split** is used to avoid patient-level data leakage.
- Labels are weakly supervised and highly imbalanced; no relabeling or curation is performed.

The dataset is not included in this repository. Summary tables of the dataset is provided in /data.

---

## Model and Training

- Architecture: **ResNet-50** (torchvision implementation)
- Initialization: None
- Optimizer: Adam
- Learning rate: 1e-4
- Hyperparameter tuning: none
- Number of runs: single seed

The emphasis is on establishing a reasonable baseline rather than optimizing performance.

---

## Task

- **Multilabel classification** across all pathologies in the dataset
- Each image may have multiple positive labels
- Severe class imbalance is present and left uncorrected

---

## Evaluation

Reported metrics:
- **AUROC per class**
- **Macro-average AUROC**

Accuracy is intentionally not reported due to its limited usefulness in imbalanced multilabel medical settings.

To inspect model behavior beyond scalar metrics, **Grad-CAM** visualizations are generated and averaged per class.

No out-of-distribution testing, subgroup analysis, calibration assessment, or uncertainty estimation is performed.

---

## Results (Summary)

- Per-class AUROC scores range approximately from **0.64 to 0.85**
- Performance varies substantially across pathologies
- Results are consistent with expectations for a standard CNN baseline on this dataset

Detailed metrics diagrams are saved under the `output/` directory.

---

## Interpretability Notes

Grad-CAM is used here as a **diagnostic visualization tool**, not as a definitive explanation of model reasoning.

Important caveats:
- Grad-CAM highlights regions correlated with predictions, not causal evidence
- Explanations depend on model architecture and training distribution
- No stability or robustness analysis of explanations is conducted

Visualizations should therefore be interpreted with caution.

---

## Limitations

This project has several explicit limitations:
- Single training run (no variance estimation)
- No robustness or distribution-shift evaluation
- No explicit handling of class imbalance
- No calibration or uncertainty analysis
- Known dataset biases and label noise remain unaddressed
- No clinical claims or validation

These limitations define the scope of what can be concluded from the results.

---

## What This Project Is / Is Not

**This project is:**
- a clean baseline implementation,
- a demonstration of appropriate evaluation practices,
- an exercise in careful interpretation of results.

**This project is not:**
- a state-of-the-art medical AI system,
- a robustness or interpretability study,
- a deployable or clinically validated model.

---

## Repository Structure

.
├── data/ # Dataset summary
├── scripts/ # Training scripts
├── src/ # source code
├── notebooks/ # Notebokes and Grad-CAM Analysis
├── outputs/ # Saved metrics and outputs
└── README.md


---

## Takeaway

Training models on large datasets is relatively straightforward. Understanding what their performance and explanations actually mean—and where they fail—requires careful evaluation and explicit assumptions.

This repository represents a starting point for asking those questions, not a final answer.

---