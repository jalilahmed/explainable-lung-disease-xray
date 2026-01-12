import torch

class MultiLabelAccuracy(object):
    """Computes multi-label accuracy for binary classification tasks.
    Args:
        threshold (float): Threshold for converting probabilities to binary predictions.
        reduction (str): Reduction method to apply to the output: 'none' | 'mean'.
    """
    def __init__(self, threshold=0.5, reduction='mean'):
        """Initialize the MultiLabelAccuracy metric."""
        self.threshold = threshold
        self.reduction = reduction

    def __call__(self, logits, target):
        """Compute the multi-label accuracy.
        Args:
            logits (torch.Tensor): Predictions from the model (logits).
            target (torch.Tensor): Ground truth binary labels.
        """

        assert logits.shape == target.shape, \
            "logits and Target Tensor don't have correct shape"

        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).long()
        
        target = target.long()

        correct = (preds == target).float()  # [B, C]

        # per-sample accuracy
        acc = correct.mean(dim=1)  # [B]

        if self.reduction == 'none':
            return acc
        elif self.reduction == 'mean':
            return acc.mean()
        else:
            raise ValueError("reduction value not specified")
