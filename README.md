# Strawberry Size Classification: EfficientDet

EfficientDet part of the team study **"Strawberry Size Classification: Comparative Analysis of YOLO, Faster R-CNN and EfficientDet"**. Main team repository: https://github.com/tolgaguclu17/strawberry-grading-comparison

The detector classifies strawberries into three size classes (**Short / Medium / Long**, ids 0/1/2). The size classes come from ImageJ-calibrated length measurements. The data is the same 427-image dataset and the same Train/Val/Test split (341/43/43, seed 42) used for YOLO, so the three models can be compared directly.

## Contents

| Path | Description |
|---|---|
| `notebooks/EfficientDet_Strawberry.ipynb` | Training, hyperparameter optimisation, test evaluation, export (Colab, GPU) |
| `local/` | Same notebook for a local Windows NVIDIA GPU, with setup scripts |
| `docs/literature_review.md` | Background and references for every design decision |
| `docs/experiment_log.md` | Log of all experiments, including failed and discarded runs |
| `requirements.txt` | Library versions |

## Method summary

- **Model:** EfficientDet-D0 via the `effdet` library, trained **from scratch** (random initialisation, no pretrained backbone), the same policy as YOLO and Faster R-CNN
- **Input:** aspect-ratio-preserving letterbox resize. Stretching is avoided because the classes depend on length
- **Training:** AdamW, warm-up + cosine learning rate, gradient clipping, early stopping on validation mAP50-95
- **Hyperparameter optimisation:** staged search with a stated hypothesis per stage:
  - E1: learning rate (3e-4, 1e-3, 3e-3)
  - E3: augmentation, testing whether size-altering augmentation hurts length-based classes
- **Evaluation:**
  - COCO mAP50 and mAP50-95
  - Precision, Recall and F1 at IoU 0.5
  - confusion matrix and F1-confidence curve
  - inference time (ms/image, FPS)
- **Robustness:** final configuration trained with 3 seeds (mean ± std)
- **Result (validation, 3 seeds):** mAP50 0.977 ± 0.005, mAP50-95 0.940 ± 0.018, F1 0.811 ± 0.043
- **Output for the joint evaluation:** `effdet_test_predictions.json`, COCO-style detections in original image pixels

## Results (validation set, 43 images)

| Run | Learning rate | Augmentation | Seed | val mAP50 | val mAP50-95 | val F1 |
|---|---|---|---|---|---|---|
| E0 | 1e-3 | flip | 42 | 0.974 | 0.913 | 0.732 |
| E1_0 | 3e-4 | flip | 42 | 0.961 | 0.862 | 0.708 |
| **E1_1** (final model) | **3e-3** | **flip** | 42 | **0.983** | **0.954** | **0.807** |
| E3_scale | 3e-3 | flip + scale (interrupted, epoch 76) | 42 | 0.963 | 0.925 | 0.784 |
| FINAL_s7 | 3e-3 | flip | 7 | 0.973 | 0.947 | 0.771 |
| FINAL_s2024 | 3e-3 | flip | 2024 | 0.976 | 0.920 | 0.856 |
| **Final configuration, 3 seeds** | | | | **0.977 ± 0.005** | **0.940 ± 0.018** | **0.811 ± 0.043** |

Test-set results are produced once, for all models together, in the joint evaluation (Sections 6–8 of the team notebook).

## Files for the report

| What | Where |
|---|---|
| Method, protocol and related work | `notebooks/EfficientDet_Strawberry.ipynb` (Sections 4.1–4.6), `docs/literature_review.md` |
| All runs, decisions and process notes (including failed and interrupted runs) | `docs/experiment_log.md` |
| Per-run summary table | `results/experiment_log.csv` |
| Training curves per run | `results/runs/<run>/curves.png`, `results/runs/<run>/history.csv` |
| Final model (seed 42) | `results/runs/E1_1/best.pt`, `results/runs/E1_1/config.json` |
| Train / val / test file lists | `splits.json` |
| Hardware and library versions | `results/environment.json` |

## How to run

1. Open the notebook in Google Colab with a GPU runtime.
2. Run the data split from the main repository (Section 1.1) in the same Colab session. It creates `/content/yolo_dataset/{images,labels}/{train,val,test}`, the paths in `data.yaml`, which is the default `DATA_ROOT`.
3. Run the cells in order. After each experiment stage, update `BEST` from `experiment_log.csv`.
4. Run the test-set section (4.7) **once**, with the final configuration.

All results (logs, curves, checkpoints, metrics, predictions) are written to `OUT_DIR` on Google Drive.
