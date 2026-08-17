# Evaluation Report

## Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.5455 |
| Macro F1 | 0.5486 |
| Weighted F1 | 0.5536 |

## Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| Crypturellus cinereus | 1.0000 | 0.6000 | 0.7500 | 5 |
| Nothocercus julius | 0.3333 | 0.6000 | 0.4286 | 5 |
| Tinamus guttatus | 0.6000 | 0.7500 | 0.6667 | 4 |
| Crypturellus bartletti | 0.6667 | 0.5000 | 0.5714 | 4 |
| Crypturellus parvirostris | 0.5000 | 0.5000 | 0.5000 | 4 |
| Crypturellus variegatus | 0.4286 | 0.7500 | 0.5455 | 4 |
| Crypturellus strigulosus | 0.7500 | 0.6000 | 0.6667 | 5 |
| Crypturellus transfasciatus | 0.3333 | 0.2500 | 0.2857 | 4 |
| Ortalis columbiana | 1.0000 | 0.4000 | 0.5714 | 5 |
| Taoniscus nanus | 0.5000 | 0.5000 | 0.5000 | 4 |

## Top Misclassifications

1. Crypturellus transfasciatus → Nothocercus julius (2 times)
2. Crypturellus cinereus → Nothocercus julius (1 times)
3. Crypturellus cinereus → Taoniscus nanus (1 times)
4. Nothocercus julius → Tinamus guttatus (1 times)
5. Nothocercus julius → Crypturellus transfasciatus (1 times)

## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

## Per-Class Metrics Visualization

![Per-Class Metrics](per_class_metrics.png)
