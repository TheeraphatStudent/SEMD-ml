# Model Evaluation

The refactored evaluation pipeline reports metrics from the held-out test split and cross-validation.

## Classification metrics

- Accuracy
- Malicious-class precision
- Malicious-class recall
- Malicious-class F1
- Macro precision
- Macro recall
- Macro F1
- ROC-AUC when probability scores are available
- False-positive rate
- False-negative rate
- Confusion matrix

## Operational metrics

- Training duration in seconds
- Prediction latency in milliseconds
- Cross-validation mean
- Cross-validation standard deviation

## Model selection

- Every configured algorithm is trained on the train split.
- Validation malicious-class F1 is used for model selection.
- Test metrics are reported separately from validation metrics.

## Leakage controls

- Dataset splitting happens before scaling or balancing.
- Scaling is fit on the training fold only.
- Balancing is applied to the training fold only.
- Cross-validation uses the same leak-safe pipeline as normal training.
