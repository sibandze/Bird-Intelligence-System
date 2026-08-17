# Evaluation Report

## Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.6818 |
| Macro F1 | 0.6868 |
| Weighted F1 | 0.6865 |

## Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| Crypturellus cinereus | 1.0000 | 0.6000 | 0.7500 | 5 |
| Nothocercus julius | 0.4444 | 0.8000 | 0.5714 | 5 |
| Tinamus guttatus | 0.7500 | 0.7500 | 0.7500 | 4 |
| Crypturellus bartletti | 1.0000 | 1.0000 | 1.0000 | 4 |
| Crypturellus parvirostris | 0.6667 | 0.5000 | 0.5714 | 4 |
| Crypturellus variegatus | 0.6667 | 0.5000 | 0.5714 | 4 |
| Crypturellus strigulosus | 0.5000 | 0.8000 | 0.6154 | 5 |
| Crypturellus transfasciatus | 1.0000 | 0.5000 | 0.6667 | 4 |
| Ortalis columbiana | 0.8000 | 0.8000 | 0.8000 | 5 |
| Taoniscus nanus | 0.6667 | 0.5000 | 0.5714 | 4 |

## Top Misclassifications

1. Crypturellus cinereus → Nothocercus julius (2 times)
2. Crypturellus transfasciatus → Crypturellus strigulosus (2 times)
3. Nothocercus julius → Tinamus guttatus (1 times)
4. Tinamus guttatus → Crypturellus strigulosus (1 times)
5. Crypturellus parvirostris → Nothocercus julius (1 times)

## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

## Per-Class Metrics Visualization

![Per-Class Metrics](per_class_metrics.png)
