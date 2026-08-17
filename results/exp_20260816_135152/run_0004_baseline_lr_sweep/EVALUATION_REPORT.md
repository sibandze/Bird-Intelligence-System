# Evaluation Report

## Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4773 |
| Macro F1 | 0.4530 |
| Weighted F1 | 0.4510 |

## Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| Crypturellus cinereus | 0.5000 | 0.4000 | 0.4444 | 5 |
| Nothocercus julius | 0.5000 | 0.2000 | 0.2857 | 5 |
| Tinamus guttatus | 0.0000 | 0.0000 | 0.0000 | 4 |
| Crypturellus bartletti | 1.0000 | 0.2500 | 0.4000 | 4 |
| Crypturellus parvirostris | 0.3333 | 1.0000 | 0.5000 | 4 |
| Crypturellus variegatus | 1.0000 | 0.5000 | 0.6667 | 4 |
| Crypturellus strigulosus | 0.4000 | 0.8000 | 0.5333 | 5 |
| Crypturellus transfasciatus | 1.0000 | 0.5000 | 0.6667 | 4 |
| Ortalis columbiana | 0.3750 | 0.6000 | 0.4615 | 5 |
| Taoniscus nanus | 0.6667 | 0.5000 | 0.5714 | 4 |

## Top Misclassifications

1. Nothocercus julius → Crypturellus parvirostris (3 times)
2. Tinamus guttatus → Crypturellus strigulosus (2 times)
3. Tinamus guttatus → Ortalis columbiana (2 times)
4. Crypturellus bartletti → Crypturellus cinereus (2 times)
5. Taoniscus nanus → Crypturellus parvirostris (2 times)

## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

## Per-Class Metrics Visualization

![Per-Class Metrics](per_class_metrics.png)
