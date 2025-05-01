from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.add_state('true_positives', default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx='sum')
        self.add_state('false_positives', default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx='sum')
        self.add_state('false_negatives', default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx='sum')

    def update(self, preds, target):
        # preds: (B x C) tensor
        # target: (B,) tensor
        # print(preds.shape, target.shape)

        # 1. 예측에서 확신 높은 클래스 뽑기
        preds = torch.argmax(preds, dim=1)  # (B,)

        # 2. preds, target shape 같아야 함
        if preds.shape != target.shape:
            raise ValueError(f"Shape mismatch: preds shape {preds.shape}, target shape {target.shape}")

        for class_idx in range(self.num_classes):
            pred_is_class = preds == class_idx
            target_is_class = target == class_idx

            self.true_positives[class_idx] += torch.sum(pred_is_class & target_is_class)
            self.false_positives[class_idx] += torch.sum(pred_is_class & (~target_is_class))
            self.false_negatives[class_idx] += torch.sum((~pred_is_class) & target_is_class)

    def compute(self):
        # precision = TP / (TP + FP)
        precision = self.true_positives.float() / (self.true_positives + self.false_positives).float().clamp(min=1)
        # recall = TP / (TP + FN)
        recall = self.true_positives.float() / (self.true_positives + self.false_negatives).float().clamp(min=1)

        # F1 = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall).clamp(min=1e-8)

        # 클래스별 F1이 f1 (size = num_classes)
        # 전체 macro F1은 평균
        macro_f1 = f1.mean()

        return macro_f1

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)


        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            raise ValueError(f"Shape mismatch: preds shape {preds.shape}, target shape {target.shape}")

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
