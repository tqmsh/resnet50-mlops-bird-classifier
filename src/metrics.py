import torch
import numpy as np
import time
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    average_precision_score, balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, cohen_kappa_score, hamming_loss, jaccard_score
)

class MetricsCalculator:
    def __init__(self, num_classes: int, config: Dict[str, Any]):
        self.num_classes = num_classes
        self.metrics_config = config.get('metrics', {})
        self.inference_times = []
        self.gradient_norms = []
        self.weight_norms = []
        self.learning_rates = []

        # Use default metrics if none specified
        if not self.metrics_config:
            self.metrics_config = self._get_default_metrics_config()

    def _get_default_metrics_config(self) -> Dict[str, Any]:
        """Fallback default metrics configuration."""
        return {
            'accuracy': True, 'precision': True, 'recall': True, 'f1_score': True,
            'loss': True, 'roc_auc': True, 'top_k_accuracy': True,
            'top_k_values': [1, 3, 5]
        }

    def calculate_all_metrics(self, outputs: List[torch.Tensor], targets: List[torch.Tensor],
                            loss: Optional[float] = None, learning_rate: Optional[float] = None,
                            model: Optional[torch.nn.Module] = None) -> Dict[str, float]:
        """
        Calculate all enabled metrics based on configuration.

        Args:
            outputs: List of model predictions (logits) from accumulated batches
            targets: List of ground truth labels from accumulated batches
            loss: Current training loss
            learning_rate: Current learning rate
            model: Model for complexity metrics

        Returns:
            Dictionary of calculated metrics
        """
        # Concatenate all batches
        outputs_tensor = torch.cat(outputs, dim=0)
        targets_tensor = torch.cat(targets, dim=0)

        # Convert to numpy for sklearn metrics
        outputs_np = outputs_tensor.cpu().numpy()
        targets_np = targets_tensor.cpu().numpy()

        # Get predicted classes
        preds_np = np.argmax(outputs_np, axis=1)

        metrics = {}

        # Basic Classification Metrics
        if self.metrics_config.get('accuracy', True):
            metrics['accuracy'] = accuracy_score(targets_np, preds_np)

        if self.metrics_config.get('precision', True) or self.metrics_config.get('recall', True) or self.metrics_config.get('f1_score', True):
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets_np, preds_np, average='macro', zero_division=0
            )
            if self.metrics_config.get('precision', True):
                metrics['precision'] = precision
            if self.metrics_config.get('recall', True):
                metrics['recall'] = recall
            if self.metrics_config.get('f1_score', True):
                metrics['f1_score'] = f1

        # Advanced Classification Metrics
        if self.metrics_config.get('roc_auc', True):
            try:
                # Convert logits to probabilities using softmax
                outputs_prob = torch.softmax(outputs_tensor, dim=1).cpu().numpy()

                # Handle multi-class ROC AUC
                if self.num_classes == 2:
                    metrics['roc_auc'] = roc_auc_score(targets_np, outputs_prob[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(targets_np, outputs_prob, multi_class='ovr', average='macro')
            except Exception as e:
                metrics['roc_auc'] = 0.0
                print(f"Warning: ROC-AUC calculation failed: {e}")

        if self.metrics_config.get('average_precision', True):
            try:
                # Convert logits to probabilities using softmax
                outputs_prob = torch.softmax(outputs_tensor, dim=1).cpu().numpy()

                if self.num_classes == 2:
                    metrics['average_precision'] = average_precision_score(targets_np, outputs_prob[:, 1])
                else:
                    metrics['average_precision'] = average_precision_score(targets_np, outputs_prob, average='macro')
            except Exception as e:
                metrics['average_precision'] = 0.0
                print(f"Warning: Average Precision calculation failed: {e}")

        if self.metrics_config.get('balanced_accuracy', True):
            metrics['balanced_accuracy'] = balanced_accuracy_score(targets_np, preds_np)

        if self.metrics_config.get('matthews_corrcoef', True):
            metrics['matthews_corrcoef'] = matthews_corrcoef(targets_np, preds_np)

        # Confusion Matrix Based Metrics
        if self.metrics_config.get('specificity', True) or self.metrics_config.get('sensitivity', True) or \
           self.metrics_config.get('false_positive_rate', True) or self.metrics_config.get('false_negative_rate', True):
            cm = confusion_matrix(targets_np, preds_np)
            if cm.shape == (2, 2):  # Binary case
                tn, fp, fn, tp = cm.ravel()
                if self.metrics_config.get('specificity', True):
                    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                if self.metrics_config.get('sensitivity', True):
                    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                if self.metrics_config.get('false_positive_rate', True):
                    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                if self.metrics_config.get('false_negative_rate', True):
                    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Statistical Metrics
        if self.metrics_config.get('cohens_kappa', True):
            metrics['cohens_kappa'] = cohen_kappa_score(targets_np, preds_np)

        if self.metrics_config.get('hamming_loss', True):
            metrics['hamming_loss'] = hamming_loss(targets_np, preds_np)

        if self.metrics_config.get('jaccard_score', True):
            metrics['jaccard_score'] = jaccard_score(targets_np, preds_np, average='macro')

        # Top-K Accuracy
        if self.metrics_config.get('top_k_accuracy', True):
            top_k_values = self.metrics_config.get('top_k_values', [1, 3, 5])
            for k in top_k_values:
                if k <= self.num_classes:
                    metrics[f'top_{k}_accuracy'] = self._calculate_top_k_accuracy(outputs_np, targets_np, k)

        # Training-specific Metrics
        if loss is not None and self.metrics_config.get('loss', True):
            metrics['loss'] = loss

        if learning_rate is not None and self.metrics_config.get('learning_rate', True):
            metrics['learning_rate'] = learning_rate

        # Model Complexity Metrics
        if model is not None:
            if self.metrics_config.get('model_parameters', True):
                metrics['model_parameters'] = sum(p.numel() for p in model.parameters())

            if self.metrics_config.get('weight_norm', True):
                weight_norm = sum(p.norm().item() for p in model.parameters())
                metrics['weight_norm'] = weight_norm

        return metrics

    def _calculate_top_k_accuracy(self, outputs: np.ndarray, targets: np.ndarray, k: int) -> float:
        """Calculate top-k accuracy."""
        top_k_preds = np.argsort(outputs, axis=1)[:, -k:]
        correct = 0
        for i, target in enumerate(targets):
            if target in top_k_preds[i]:
                correct += 1
        return correct / len(targets)

    def get_enabled_metrics_list(self) -> List[str]:
        """Get list of currently enabled metrics for user display."""
        enabled = []
        for metric, is_enabled in self.metrics_config.items():
            if is_enabled and metric not in ['top_k_values']:
                enabled.append(metric.replace('_', ' ').title())
        return enabled

    def print_enabled_metrics(self):
        """Print user-friendly list of enabled metrics."""
        enabled = self.get_enabled_metrics_list()
        print("Enabled Metrics:")
        print("=" * 40)
        for i, metric in enumerate(enabled, 1):
            print(f"  {i:2d}. {metric}")
        print("=" * 40)
        print(f"Total: {len(enabled)} metrics tracked")

    def track_inference_time(self, time_ms: float):
        """Track inference time for performance monitoring."""
        if self.metrics_config.get('inference_time', True):
            self.inference_times.append(time_ms)

    def get_inference_stats(self) -> Dict[str, float]:
        """Get inference time statistics."""
        if not self.inference_times:
            return {}

        return {
            'inference_time_mean': np.mean(self.inference_times),
            'inference_time_std': np.std(self.inference_times),
            'inference_time_min': np.min(self.inference_times),
            'inference_time_max': np.max(self.inference_times)
        }