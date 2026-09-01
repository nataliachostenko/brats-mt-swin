import torch
import numpy as np
from sklearn.metrics import average_precision_score
from tools.eval_pr_metrics import convert_to_regions, convert_to_region_probs, convert_to_region_binary

def test_pr_metrics_logic():
    # 1. Test convert_to_regions on dummy ground truth
    label = torch.zeros(1, 4, 4, 4)
    label[0, 0, 0, 0] = 0
    label[0, 1, 0, 0] = 1
    label[0, 2, 0, 0] = 2
    label[0, 3, 0, 0] = 3
    
    region_labels = convert_to_regions(label)
    assert region_labels.shape == (3, 4, 4, 4), "Złe wymiary region_labels"
    
    # WT should be classes 1, 2, 3
    # TC should be classes 1, 3
    # ET should be class 3
    # WT at (0,0,0) is 0; at (1,0,0) is 1; at (2,0,0) is 1; at (3,0,0) is 1
    assert region_labels[0, 0, 0, 0] == 0
    assert region_labels[0, 1, 0, 0] == 1
    assert region_labels[0, 2, 0, 0] == 1
    assert region_labels[0, 3, 0, 0] == 1
    
    # 2. Test convert_to_region_probs
    prob = torch.zeros(4, 4, 4, 4)
    prob[0] = 0.4
    prob[1] = 0.3
    prob[2] = 0.2
    prob[3] = 0.1
    
    region_probs = convert_to_region_probs(prob)
    assert region_probs.shape == (3, 4, 4, 4), "Złe wymiary region_probs"
    # WT prob = prob[1] + prob[2] + prob[3] = 0.6
    # TC prob = prob[1] + prob[3] = 0.4
    # ET prob = prob[3] = 0.1
    assert torch.allclose(region_probs[0], torch.tensor(0.6))
    assert torch.allclose(region_probs[1], torch.tensor(0.4))
    assert torch.allclose(region_probs[2], torch.tensor(0.1))
    
    # 3. Test convert_to_region_binary
    region_binary = convert_to_region_binary(prob)
    assert region_binary.shape == (3, 4, 4, 4), "Złe wymiary region_binary"
    # since prob[0] = 0.4 is max, argmax is 0. So all region_binary should be 0.
    assert torch.all(region_binary == 0)
    
    # Let's change probs to make class 1 max
    prob_class1 = torch.zeros(4, 4, 4, 4)
    prob_class1[1] = 1.0
    region_binary_class1 = convert_to_region_binary(prob_class1)
    # class 1: WT=1, TC=1, ET=0
    assert torch.all(region_binary_class1[0] == 1)
    assert torch.all(region_binary_class1[1] == 1)
    assert torch.all(region_binary_class1[2] == 0)
    
    # 4. Test metric calculation logic
    y_true = np.array([0, 1, 1, 1])
    y_score = np.array([0.1, 0.9, 0.8, 0.7])
    ap = average_precision_score(y_true, y_score)
    assert ap == 1.0

    # 5. Test MONAI DiceMetric and HausdorffDistanceMetric
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
    dice_m = DiceMetric(include_background=True, reduction="none")
    hd_m = HausdorffDistanceMetric(percentile=95, include_background=True, reduction="none")

    # dummy predictions and labels (Batch=1, Channel=3, H=4, W=4, D=4)
    pred_dummy = region_binary.unsqueeze(0)
    label_dummy = region_labels.unsqueeze(0)

    dice_val = dice_m(pred_dummy, label_dummy)
    hd_val = hd_m(pred_dummy, label_dummy)

    assert dice_val.shape == (1, 3)
    assert hd_val.shape == (1, 3)
    
    print("Test logic and MONAI metrics passed successfully!")
