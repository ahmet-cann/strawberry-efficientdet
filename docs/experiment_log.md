# Experiment Log: EfficientDet

Each row comes from `experiment_log.csv` (written automatically by the notebook). **Failed, crashed and discarded runs stay in the log.** A failed attempt with an explanation is part of the result.

Selection metric: validation mAP50-95. Differences below 0.01 are treated as ties, and the simpler or faster setting is preferred.

## Runs

| ID | Date | Change vs. previous best | Hypothesis | Best epoch | val mAP50 | val mAP50-95 | val F1 | Decision |
|---|---|---|---|---|---|---|---|---|
| E0 | 20.10.2026 | Baseline: D0, 512 px, random init, lr 1e-3, flip aug | Reference | 66 / 86 | 0.974 | 0.913 | 0.732 | Reference for E1 |
| E1_0 | 20.10.2026 | lr = 3e-4 | Lower lr is more stable but may under-fit from scratch | 79 / 99 | 0.961 | 0.862 | 0.708 | Rejected (−0.051 vs E0) |
| E1_1 | 20.10.2026 | lr = 3e-3 | Higher lr converges faster but may diverge | 83 / 100 | 0.983 | 0.954 | 0.807 | **Accepted** (+0.041 vs E0) → `BEST['lr'] = 3e-3` |
| E3_scale | 22.10.2026 | + random scale 0.7–1.3 (lr 3e-3) | Changes apparent size, expected to hurt Short/Medium/Long | 76 / 76 (interrupted) | 0.963 | 0.925 | 0.784 | Rejected: below E1_1 at every matched epoch (mean ep 71–76: 0.908 vs 0.938) → keep `flip` |
| FINAL_s7 | 22.10.2026 | Final config, seed 7 (Colab T4) | Variance across seeds | 95 / 100 | 0.973 | 0.947 | 0.771 | – |
| FINAL_s2024 | 23.10.2026 | Final config, seed 2024 (RTX 4060) | Variance across seeds | 42 / 62 | 0.976 | 0.920 | 0.856 | – |
| **FINAL (3 seeds)** | | Seeds 42 (= E1_1) / 7 / 2024 | | | **0.977 ± 0.005** | **0.940 ± 0.018** | **0.811 ± 0.043** | Reported result (mean ± sample std) |

## Process notes

Write one entry per issue: **what we tried → what happened → why → what we changed.**

### Problems and fixes
- **Non-deterministic split.** The original split cell builds the file list with `os.listdir`, whose order is not guaranteed, so `random_state=42` produced different sets on different machines (YOLO run: val Short/Medium/Long = 16/14/13; new session: 18/13/12). The file list is now sorted before splitting and the split is saved as `splits.json` on Drive and shared with the team, so all three models use the same train/val/test images.
- **Dataset layout.** The labelled data is a single folder (images + LabelImg `.txt` files + `classes.txt`), not separate `images/` and `labels/` folders. The split cell was adapted to read this folder directly from Drive.
- **Training speed on free Colab (Tesla T4).** The first E0 attempt ran at ~50 s/epoch; the bottleneck is decoding the ~4.4 MB original JPEGs on 2 CPU cores, not the GPU. Pre-resizing the images was considered and rejected so that the original images are used unchanged. To fit the compute budget, the schedule was shortened from 300 to 100 epochs and early-stopping patience from 50 to 20 epochs; the first attempt was stopped at epoch 9 (val mAP50 0.878) and E0 was restarted with the new schedule.
- **Switch to a local GPU.** After E1 the free Colab session limit (~5.5 h) and daily GPU quota made the remaining runs impractical. Subsequent runs use a local NVIDIA RTX 4060 Laptop GPU (8 GB) with the same code, split and settings (`local/EfficientDet_Strawberry_local.ipynb`); only training times differ.
- **Hardware difference.** YOLO was trained on an NVIDIA RTX PRO 6000 (8 data-loader workers); EfficientDet on a Tesla T4 (2 CPU cores). Training times are therefore not comparable across models, and inference speed must be measured for all models on the same GPU in the joint evaluation.

### Observations per run
- **E0 (73.5 min).** Val mAP50 reached 0.87 by epoch 10 and plateaued after ~epoch 60 (mAP50-95 ≈ 0.88–0.91). Val loss rose slightly in the last 20 epochs (0.145 → 0.174) and early stopping ended training at epoch 86 (best epoch 66). Precision (0.65) is clearly lower than recall (0.85): the model often adds a second box on the same strawberry with a neighbouring size class, which points to uncertainty at the class boundaries. The best value (0.913) is partly a fluctuation; neighbouring epochs are 0.88–0.90, which is expected with 43 validation images and motivates the multi-seed evaluation.
- **E1_0, lr = 3e-4 (88.7 min).** Learning was slower than in E0: val mAP50-95 stayed at 0 for 4 epochs and reached 0.70 only at epoch 24 (E0: epoch 15). Training ran almost to the 100-epoch limit (early stop at 99, best epoch 79) and plateaued at mAP50-95 ≈ 0.85, below E0 (0.913). The hypothesis that a lower learning rate under-fits when training from scratch is supported: with the 100-epoch budget, lr = 3e-4 does not converge as far as lr = 1e-3.
- **E1_1, lr = 3e-3 (86.0 min).** Fastest convergence of the three learning rates (val mAP50 0.82 at epoch 4) with no sign of divergence. Val mAP50-95 rose steadily to ≈ 0.95 and was still improving slowly when the 100-epoch limit was reached (no early stop; best epoch 83), so a longer schedule might add a little more. Precision (0.73) and F1 (0.81) are also the highest so far, i.e. fewer duplicate boxes with a neighbouring size class. Result: lr = 3e-3 is selected for the next stages. The hypothesis that a higher learning rate may diverge is rejected for this setting (AdamW with warm-up and gradient clipping).
- **E3_scale, first attempt (local RTX 4060, aborted).** Started with lr = 1e-3 while E1_1 was still running on Colab. Stopped at epoch 31 (val mAP50-95 0.821) with an out-of-memory error in system RAM: 8 persistent data-loader workers for training plus 8 for validation (16 processes, each loading PyTorch on Windows) exceeded the available memory. The number of workers was reduced (4 for training, 2 for validation). The run is discarded and E3_scale is repeated with the selected lr = 3e-3, so that it differs from the reference (E1_1) only in augmentation.
- **E3_scale, lr = 3e-3 (Colab T4, interrupted at epoch 76).** Same settings as E1_1 plus random scaling (0.7–1.3). The run stopped at epoch 76 of 100 (KeyboardInterrupt) before the end of the cosine schedule, so the summary row and `history.csv` were not written; the values above are taken from the per-epoch output, and the best checkpoint (epoch 76) is on Drive. Compared epoch by epoch with E1_1, scale augmentation was lower throughout: mean val mAP50-95 over epochs 71–76 was 0.908 vs 0.938, and the training loss stayed much higher (0.11 vs 0.04 at epoch 76), i.e. the task became harder rather than the model generalising better. Because the gap is consistent over the whole run, the comparison was judged sufficient and the run was not repeated (compute budget). Conclusion: size-altering augmentation does not help when the class is defined by absolute length, and `flip` is kept.
- **FINAL_s2024 (local RTX 4060, 26.6 min).** Final configuration (D0, lr 3e-3, flip) with seed 2024. Early stopping ended training at epoch 62 (best epoch 42): val mAP50 0.976, mAP50-95 0.920, P 0.783, R 0.947, F1 0.856. mAP50-95 is 0.034 lower than seed 42 (E1_1), but F1 is higher. Note that with patience 20 this run stopped while the learning rate was still high (cosine schedule at epoch 62 of 100), whereas E1_1 ran the full schedule; part of the seed-to-seed difference is therefore due to early stopping rather than to the initialisation alone. Training speed on the RTX 4060 with 4 data-loader workers was ~26 s/epoch, about twice as fast as the Colab T4.
- **FINAL_s7 (Colab T4, 94.4 min).** Full 100-epoch schedule, best epoch 95: val mAP50 0.973, mAP50-95 0.947, F1 0.771, i.e. within 0.007 mAP50-95 of seed 42. The Colab cell was set to run seeds 7 and 2024 in sequence; seed 2024 had already been completed on the local GPU, so the Colab copy was stopped at epoch 3 and discarded. Because the summary is printed only after the loop, the seed-7 row was read from `experiment_log.csv`.
- **Seed robustness.** Over three seeds the final configuration gives val mAP50 0.977 ± 0.005 and mAP50-95 0.940 ± 0.018 (mean ± sample std). Localisation (mAP50) is very stable; the stricter mAP50-95 and F1 vary more, and the lowest value comes from the run that stopped early (seed 2024). With only 43 validation images these spreads are expected, and the mean ± std, not the single best run, is the figure to compare with the other models.

### Hypotheses confirmed or rejected
- **E3 scale augmentation — supported (with caveat).** Random rescaling lowered val mAP50-95 by about 0.03 relative to E1_1 at matched epochs. This is consistent with the hypothesis that changing the apparent size of the fruit contradicts length-based labels. Caveat: the run was interrupted at epoch 76/100 and per-class values were not computed, so the effect on the Medium class specifically could not be isolated.

### Data checks
- Dataset: 427 images, 427 label files, no missing labels; exactly one strawberry per image. Class instances: 0 (Short) 143, 1 (Medium) 141, 2 (Long) 143. `classes.txt` contains only the ids `0 1 2`; names follow `data.yaml` (to be confirmed by the data owner).
- Class distribution per split (train / val / test): Short 113 / 18 / 12, Medium 113 / 13 / 15, Long 115 / 12 / 16; images 341 / 43 / 43.
- Sanity-check image: boxes aligned with the fruit after augmentation and letterbox.
- Leakage check: no file name appears in more than one split.

## Final configuration

| Parameter | Value |
|---|---|
| Variant / input size | EfficientDet-D0, 512 × 512 (letterbox, grey padding 114) |
| Initialisation | random (team policy) |
| Learning rate / weight decay | 3e-3 (AdamW, 5-epoch warm-up, cosine decay) / 1e-4 |
| Batch size | 8 |
| Augmentation | flip (horizontal + vertical) + brightness/contrast + hue/saturation; no scaling |
| Epochs (early stopped at) | max 100, early-stopping patience 20 on val mAP50-95 |
| GPU / library versions | Tesla T4 (Colab: E0, E1, E3, FINAL_s7) and RTX 4060 Laptop (FINAL_s2024); see `environment.json` |
| Delivered model | `runs/E1_1/best.pt` (seed 42, val mAP50-95 0.954), on Drive in `strawberry_effdet/` |

## Test results (run once, by the evaluation member)

| Class | P | R | F1 | mAP50-95 |
|---|---|---|---|---|
| all | | | | |
| Short | | | | |
| Medium | | | | |
| Long | | | | |

Overall mAP50: · Inference: ms/img, FPS (GPU, batch 1, input px) · Parameters: M
