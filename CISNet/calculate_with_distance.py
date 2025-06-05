import os
import re
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def turn_colors_to_class_labels_zones(mask):
    mask_class_labels = np.copy(mask)
    mask_class_labels[mask == 0] = 0
    mask_class_labels[mask == 64] = 1
    mask_class_labels[mask == 127] = 2
    mask_class_labels[mask == 254] = 3
    return mask_class_labels

def get_matching_out_of_folder(file_name, folder):
    files = os.listdir(folder)
    matching_files = [a for a in files if re.match(pattern=os.path.split(file_name)[1][:-4], string=os.path.split(a)[1])]
    if len(matching_files) > 1:
        print("Something went wrong!")
        print(f"targets_matching: {matching_files}")
    if len(matching_files) < 1:
        print("Something went wrong! No matches found")
    return matching_files[0]

def multi_class_metric(metric_function, predicted, target):
    labels = [0, 1, 2, 3]
    y_true = np.concatenate([t.flatten() for t in target])
    y_pred = np.concatenate([p.flatten() for p in predicted])
    scores = metric_function(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0
    )
    supports = [(y_true == l).sum() for l in labels]
    total_support = sum(supports)
    macro = np.sum([s * sc for s, sc in zip(supports, scores)]) / total_support if total_support > 0 else 0
    metric_na, metric_stone, metric_glacier, metric_ocean = scores
    return [macro, metric_na, metric_stone, metric_glacier, metric_ocean]

def calculate_segmentation_metrics_front_distance(complete_predicted_masks, complete_directory,
                                              directory_of_complete_targets, front_masks_directory, distance = 1000):
    print("Calculate segmentation metrics around curve masks (with {} distance)...\n\n".format(distance))
    all_predicted_pixels = []
    all_target_pixels = []
    for file_name in complete_predicted_masks:
        print(f"File: {file_name}")
        complete_predicted_mask = cv2.imread(os.path.join(complete_directory, file_name), cv2.IMREAD_GRAYSCALE)
        matching_target_file = get_matching_out_of_folder(file_name, directory_of_complete_targets)
        complete_target = cv2.imread(os.path.join(directory_of_complete_targets, matching_target_file),
                                     cv2.IMREAD_GRAYSCALE)
        matching_front_file = get_matching_out_of_folder(file_name, front_masks_directory)

        predicted_labels = turn_colors_to_class_labels_zones(complete_predicted_mask)
        target_labels = turn_colors_to_class_labels_zones(complete_target)

        front_mask = cv2.imread(os.path.join(front_masks_directory, matching_front_file), cv2.IMREAD_GRAYSCALE)
        front_mask = (front_mask > 0).astype(np.uint8)

        kernel_size = (int)(distance / int(file_name.split('_')[3]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1))
        dilated_mask = cv2.dilate(front_mask, kernel)
        predicted_region = predicted_labels[dilated_mask == 1]
        target_region = target_labels[dilated_mask == 1]

        if len(predicted_region) == 0:
            print(f"Warning: No pixels found in dilated region for {file_name}, skipping.")
            continue

        all_predicted_pixels.append(predicted_region)
        all_target_pixels.append(target_region)

    iou = multi_class_metric(jaccard_score, all_predicted_pixels, all_target_pixels)
    precision = multi_class_metric(precision_score, all_predicted_pixels, all_target_pixels)
    recall = multi_class_metric(recall_score, all_predicted_pixels, all_target_pixels)
    f1 = multi_class_metric(f1_score, all_predicted_pixels, all_target_pixels)

    print("Precision", precision)
    print("Recall", recall)
    print("F1 Score", f1)
    print("IoU", iou)