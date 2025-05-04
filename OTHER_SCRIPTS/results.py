import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os


ground_truth_path = ".../ground_truth.png"
prediction_path   = ".../prediction_mask.png"
rgb_image_path    = ".../4311-15_1.png"
output_path       = "output_overlay.png"



def load_binary_mask(path):
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return (arr > 127).astype(np.uint8)


def evaluate_on_rgb_background(rgb_path, ground_truth, prediction, alpha=0.3):
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)

    if rgb_bgr is None:
        raise FileNotFoundError(f"Failed to load RGB image from {rgb_path}")

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    if rgb.shape[:2] != ground_truth.shape:
        raise ValueError(f"Dimensions do not match: RGB {rgb.shape[:2]} vs GT {ground_truth.shape}")


    TP = (ground_truth == 1) & (prediction == 1)
    FP = (ground_truth == 0) & (prediction == 1)
    FN = (ground_truth == 1) & (prediction == 0)
    TN = (ground_truth == 0) & (prediction == 0)

    tp = np.sum(TP)
    fp = np.sum(FP)
    fn = np.sum(FN)
    tn = np.sum(TN)


    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0  # Intersection over Union

    print("==== FULL IMAGE EVALUATION  ====")
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1_score:.3f}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"IoU:       {iou:.3f}")


    overlay = np.zeros_like(rgb, dtype=np.uint8)
    overlay[TP] = [0, 255, 0]       # Green
    overlay[FP] = [255, 0, 0]       # Red
    overlay[FN] = [255, 255, 0]     # Yellow
    #overlay[TN] = [0, 0, 255]      # Blue


    blended = rgb.copy()
    mask = (overlay > 0).any(axis=2)


    blended[mask] = cv2.addWeighted(rgb[mask], 1 - alpha, overlay[mask], alpha, 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(blended)
    plt.title("Results")
    plt.axis('off')


    result_image_path = os.path.join(
        os.path.dirname(ground_truth_path),
        "2classification_overlay.png"
    )
    plt.savefig(result_image_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()

    print(f"Finished!: {result_image_path}")


ground_truth = load_binary_mask(ground_truth_path)
prediction   = load_binary_mask(prediction_path)


evaluate_on_rgb_background(rgb_image_path, ground_truth, prediction, alpha=0.6)
