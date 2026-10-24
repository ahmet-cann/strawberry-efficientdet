# Literature Review: EfficientDet for Strawberry Size Classification

This review covers the background for the EfficientDet part of the study *Strawberry Size Classification: Comparative Analysis of YOLO, Faster R-CNN and EfficientDet*. It is organised around the decisions taken in the notebook, so that every design choice can be traced to a source.

## 1. Object detection families

Modern CNN detectors fall into two groups. **Two-stage** detectors such as Faster R-CNN [14] first generate region proposals and then classify and refine each proposal. They are usually accurate but slower. **One-stage** detectors such as YOLO [15] and SSD predict classes and boxes in a single pass over dense locations. They are faster, but historically less accurate, mainly because of the extreme imbalance between a few object locations and many background locations.

Lin et al. [4] identified this foreground/background imbalance as the main reason for the accuracy gap and proposed the **focal loss**, which reduces the weight of easy, well-classified examples. With the focal loss, the one-stage RetinaNet matched two-stage accuracy. EfficientDet uses the same loss.

## 2. EfficientDet

EfficientDet [1] builds on three ideas:

1. **EfficientNet backbone** [2]. EfficientNet showed that scaling network depth, width and input resolution *together*, with fixed ratios, gives better accuracy per FLOP than scaling any one of them.
2. **BiFPN.** The Feature Pyramid Network [3] combines features from different resolutions top-down so that objects of different sizes can be detected. BiFPN adds bottom-up paths and *learned weights* for each input feature, because features at different resolutions contribute unequally.
3. **Compound scaling of the whole detector.** One coefficient scales the backbone, BiFPN, heads and input resolution together, producing the D0 (512 px) to D7 family. We use the smallest member, D0 (512 px), which has a similar size to YOLOv8n.

**Relevance to this study.** EfficientDet is one-stage like YOLOv8, but differs in backbone, neck and head (anchor-based, focal loss). Including it lets the comparison separate "one-stage vs. two-stage" effects from architecture-specific effects.

## 3. Comparing detectors fairly

Huang et al. [5] compared Faster R-CNN, R-FCN and SSD under controlled conditions. They showed that the backbone, input resolution and number of proposals change speed and accuracy as much as the choice of meta-architecture. A comparison is only meaningful when these factors and the hardware are reported. We therefore report the input size, parameter count, GPU and library versions for each model, and measure inference time with the same procedure for all models.

Padilla et al. [10] showed that popular toolkits implement "mAP" differently (interpolation, IoU thresholds, handling of difficult objects), so numbers from different toolkits are not directly comparable. YOLO (Ultralytics), torchvision and effdet each report their own mAP. In this study all models are scored with the **COCO protocol** [11] (pycocotools), and EfficientDet exports its test predictions in COCO format for the joint evaluation.

## 4. Deep learning for fruit detection and grading

Deep learning is now the dominant approach for fruit detection in agriculture [16, 17]. For strawberries specifically:

- Zhou et al. [7] used an improved Faster R-CNN to detect strawberries in greenhouse images and assess their plumpness, a size- and shape-related quality trait. This is close to our task.
- Santiago et al. [6] compared SSD- and EfficientDet-based detectors for three strawberry diseases and evaluated them on an embedded NVIDIA Jetson Nano, an example of the speed/accuracy trade-off in a deployment setting.
- Cordeiro et al. [8] applied YOLOv8n to eight strawberry quality classes (defects, diseases, ripeness). They benchmarked YOLOv8 n/s/m variants, used early stopping after observing rising validation loss in long runs, and listed a comparison with **Faster R-CNN and EfficientDet** as future work. Our study addresses this directly, on a size-grading task.

**Gap.** Most strawberry studies classify visual condition (disease, ripeness). Fewer address *size grading*, and the effect of size-altering augmentation on size labels is rarely discussed.

## 5. Data augmentation and label consistency

Augmentation is the standard remedy for small datasets [9]. An augmentation is only valid if it does **not change the label**. Flips and colour changes do not change the physical length of a strawberry. Random scaling, zoom crops and mosaic change the apparent size of the fruit in pixels. Because our classes (Short/Medium/Long) are defined by length, measured with a fixed camera setup and ImageJ calibration, size-altering augmentation may produce training images whose label no longer matches their appearance. Experiment E3 tests this hypothesis.

## 6. Data leakage and reproducibility

Kapoor and Narayanan [12] reviewed leakage in ML-based science and found it to be a widespread cause of over-optimistic results. Typical forms are pre-processing on the full dataset before splitting, duplicates across splits, and repeated use of the test set for model selection. Our protocol addresses these:

- the split is made once, before augmentation
- file-name overlap between splits is checked automatically
- hyperparameters are selected on the validation set only
- the test set is evaluated once

Near-duplicate photos of the same fruit can still leak across splits, and this should be checked when interpreting very high scores.

## 7. Hyperparameter optimisation

Bergstra and Bengio [13] showed that random search is more efficient than grid search when only a few hyperparameters matter. Within our compute budget, we use a staged, hypothesis-driven search (learning rate → augmentation), followed by a three-seed evaluation of the final configuration. We report every run, including discarded ones. The optimiser is AdamW [18], with warm-up and cosine decay.

## References

1. Tan, M.; Pang, R.; Le, Q.V. EfficientDet: Scalable and Efficient Object Detection. In *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 10781–10790. arXiv:1911.09070.
2. Tan, M.; Le, Q.V. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. In *Proc. 36th Int. Conf. Machine Learning (ICML)*, PMLR 97, 2019, pp. 6105–6114.
3. Lin, T.-Y.; Dollár, P.; Girshick, R.; He, K.; Hariharan, B.; Belongie, S. Feature Pyramid Networks for Object Detection. In *Proc. CVPR*, 2017, pp. 2117–2125.
4. Lin, T.-Y.; Goyal, P.; Girshick, R.; He, K.; Dollár, P. Focal Loss for Dense Object Detection. In *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 2980–2988.
5. Huang, J.; Rathod, V.; Sun, C.; Zhu, M.; Korattikara, A.; Fathi, A.; Fischer, I.; Wojna, Z.; Song, Y.; Guadarrama, S.; Murphy, K. Speed/Accuracy Trade-Offs for Modern Convolutional Object Detectors. In *Proc. CVPR*, 2017, pp. 7310–7319.
6. Santiago, A.; Solaque, L.; Velasco, A. Strawberry Disease Detection in Precision Agriculture. In *Proc. 18th Int. Conf. Informatics in Control, Automation and Robotics (ICINCO)*, Vol. 1, 2021, pp. 537–544. doi:10.5220/0010616405370544.
7. Zhou, C.; Hu, J.; Xu, Z.; Yue, J.; Ye, H.; Yang, G. A Novel Greenhouse-Based System for the Detection and Plumpness Assessment of Strawberry Using an Improved Deep Learning Technique. *Frontiers in Plant Science* 2020, 11, 559. doi:10.3389/fpls.2020.00559.
8. Cordeiro, L.d.S.; Nääs, I.d.A.; Okano, M.T. Smart Postharvest Management of Strawberries: YOLOv8-Driven Detection of Defects, Diseases, and Maturity. *AgriEngineering* 2025, 7, 246. doi:10.3390/agriengineering7080246.
9. Shorten, C.; Khoshgoftaar, T.M. A Survey on Image Data Augmentation for Deep Learning. *Journal of Big Data* 2019, 6, 60. doi:10.1186/s40537-019-0197-0.
10. Padilla, R.; Passos, W.L.; Dias, T.L.B.; Netto, S.L.; da Silva, E.A.B. A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit. *Electronics* 2021, 10, 279. doi:10.3390/electronics10030279.
11. Lin, T.-Y.; Maire, M.; Belongie, S.; Hays, J.; Perona, P.; Ramanan, D.; Dollár, P.; Zitnick, C.L. Microsoft COCO: Common Objects in Context. In *Proc. European Conf. Computer Vision (ECCV)*, LNCS 8693, 2014, pp. 740–755.
12. Kapoor, S.; Narayanan, A. Leakage and the Reproducibility Crisis in Machine-Learning-Based Science. *Patterns* 2023, 4, 100804. doi:10.1016/j.patter.2023.100804.
13. Bergstra, J.; Bengio, Y. Random Search for Hyper-Parameter Optimization. *Journal of Machine Learning Research* 2012, 13, 281–305.
14. Ren, S.; He, K.; Girshick, R.; Sun, J. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. In *Advances in Neural Information Processing Systems (NeurIPS)* 28, 2015, pp. 91–99.
15. Redmon, J.; Divvala, S.; Girshick, R.; Farhadi, A. You Only Look Once: Unified, Real-Time Object Detection. In *Proc. CVPR*, 2016, pp. 779–788.
16. Kamilaris, A.; Prenafeta-Boldú, F.X. Deep Learning in Agriculture: A Survey. *Computers and Electronics in Agriculture* 2018, 147, 70–90. doi:10.1016/j.compag.2018.02.016.
17. Koirala, A.; Walsh, K.B.; Wang, Z.; McCarthy, C. Deep Learning – Method Overview and Review of Use for Fruit Detection and Yield Estimation. *Computers and Electronics in Agriculture* 2019, 162, 219–234. doi:10.1016/j.compag.2019.04.017.
18. Loshchilov, I.; Hutter, F. Decoupled Weight Decay Regularization. In *Int. Conf. Learning Representations (ICLR)*, 2019.

**Software:** effdet (PyTorch EfficientDet, R. Wightman), https://github.com/rwightman/efficientdet-pytorch · torchmetrics / pycocotools for COCO mAP.
