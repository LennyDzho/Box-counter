from ultralytics import YOLO
import os
import cv2
from collections import defaultdict
import math
import numpy as np
import csv

model = YOLO("best.pt")
class_names = model.names
image_root = "images"
output_root = "results/images"
os.makedirs(output_root, exist_ok=True)

report_path = "results/result.csv"
os.makedirs(os.path.dirname(report_path), exist_ok=True)


def get_objects_from_image(img_path, save_path=None, mask_only=True):
    results = model(img_path, conf=0.25)[0]
    objects = []

    if results.boxes is None:
        return objects

    masks = results.masks.data.cpu().numpy() if results.masks else [None] * len(results.boxes)

    for box, cls_id, conf, mask in zip(results.boxes.xywh, results.boxes.cls, results.boxes.conf, masks):
        class_name = class_names[int(cls_id)]
        x, y, w, h = box.tolist()
        cx, cy = round(x), round(y)
        objects.append({
            "class": class_name,
            "coord": (cx, cy),
            "box": (x, y, w, h),
            "conf": float(conf),
            "mask": mask
        })

    if save_path:
        if mask_only:
            img_masked = results.plot(boxes=False, labels=False, probs=False)
            cv2.imwrite(save_path, img_masked)
        else:
            img_annotated = results.plot(boxes=True, labels=True, probs=False)
            cv2.imwrite(save_path, img_annotated)

    return objects

def filter_nested_masks(objects, threshold=0.8):
    filtered = []
    for i, obj_a in enumerate(objects):
        mask_a = obj_a.get("mask")
        if mask_a is None:
            filtered.append(obj_a)
            continue

        area_a = mask_a.sum()
        keep = True

        for j, obj_b in enumerate(objects):
            if i == j or obj_b.get("mask") is None:
                continue
            mask_b = obj_b["mask"]
            intersection = np.logical_and(mask_a, mask_b).sum()
            if intersection / area_a > threshold:
                keep = False
                break

        if keep:
            filtered.append(obj_a)
    return filtered

def build_top_dict(objects, min_dist=80):
    top_dict = {}
    left_dict = defaultdict(list)
    counter = defaultdict(int)
    tops = []
    min_dist = {
        "tablet": 40,
        "group_box": 80,
        "laptop": 40
    }

    for obj in objects:
        x, y = obj["coord"]
        obj_class = "_".join(obj["class"].split("_")[:-1])
        if obj["class"].endswith("_top"):
            too_close = any(
                math.hypot(x - tx, y - ty) < min_dist[obj_class]
                for (_key, (tx, ty)) in tops
                if obj["class"] == _key[:len(obj["class"])]
            )
            if not too_close:
                key_base = obj["class"]
                counter[key_base] += 1
                key = f"{key_base}_{counter[key_base] - 1}"
                tops.append((key, obj["coord"]))
                top_dict[key] = {
                    "count": 0,
                    "coord": obj["coord"],
                    "box": obj["box"],
                    "conf": obj["conf"],
                    "tops_left": None,
                    "left_chain": []
                }

    left_candidates = [obj for obj in objects if obj["class"].endswith("_left")]
    filtered = filter_nested_masks(left_candidates)
    for obj in filtered:
        base_class = obj["class"].replace("_left", "")
        left_dict[base_class].append(obj)

    for top_key, top_data in top_dict.items():
        base_class = top_key.rsplit("_", 1)[0].replace("_top", "")
        x, y, w, h = top_data["box"]
        x_left = x - w / 2
        x_right = x + w / 2
        ty = y + w / 2

        used_boxes = set()

        for left_obj in left_dict.get(base_class, []):
            lx_center, ly = left_obj["coord"]
            x_, y_, w_left, h_left = left_obj["box"]
            bottom_y = ly - h_left / 2

            if bottom_y - 20 < ty and x_left < x_ < x_right:
                top_data["count"] += 1
                top_data["tops_left"] = left_obj
                top_data["left_chain"].append(left_obj)
                used_boxes.add(left_obj["box"])
                break

        current = top_data["tops_left"]
        if not current:
            continue

        while True:
            next_match = None
            cx, cy, cw, ch = current["box"]
            cx_left = cx - cw / 2
            cx_right = cx + cw / 2
            c_bottom = cy + ch / 2

            for left_obj in left_dict.get(base_class, []):
                if left_obj["box"] in used_boxes:
                    continue
                lx_center, ly = left_obj["coord"]
                _, _, w_left, h_left = left_obj["box"]
                bottom_y = ly - h_left / 2

                if bottom_y - 10 < c_bottom and cx_left < lx_center < cx_right:
                    next_match = left_obj
                    break

            if not next_match:
                break

            top_data["count"] += 1
            current = next_match
            top_data["left_chain"].append(current)
            used_boxes.add(current["box"])

    return {
        k: {
            "count": v["count"],
            "top_coord": v["box"],
            "conf": v["conf"],
            "left_chain": v["left_chain"]
        } for k, v in top_dict.items()
    }

def merge_dicts(left_dict, right_dict):
    merged = {}
    all_keys = set(left_dict.keys()).union(right_dict.keys())
    for key in all_keys:
        l_val = left_dict.get(key)
        r_val = right_dict.get(key)
        if l_val and r_val:
            if l_val["count"] == 0 and r_val["count"] > 0 and r_val["top_coord"][1] > l_val["top_coord"][1]:
                merged[key] = r_val
            else:
                merged[key] = l_val
        elif l_val:
            merged[key] = l_val
        elif r_val:
            merged[key] = r_val
    return merged

def draw_top_labels_on_image(img_path, save_path, top_dict, all_objects=None):
    img = cv2.imread(img_path)
    used_left_centers = set()
    for data in top_dict.values():
        for left in data["left_chain"]:
            used_left_centers.add(left["coord"])
    if all_objects:
        for obj in all_objects:
            if obj["class"].endswith("_left") and obj["coord"] in used_left_centers:
                x, y = obj["coord"]
                cv2.circle(img, (x, y), radius=4, color=(0, 255, 0), thickness=-1)
    for key, data in top_dict.items():
        x, y, w, h = data["top_coord"]
        cx, cy = int(x), int(y)
        conf = data.get("conf", 0)
        label = f"{key} ({conf:.2f})"
        cv2.circle(img, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)
        cv2.putText(img, label, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(save_path, img)

def process_pallet_folder(pallet_path, folder_name):
    left_path = os.path.join(pallet_path, "left.png")
    right_path = os.path.join(pallet_path, "right.png")
    left_mask_path = os.path.join(output_root, f"{folder_name}_left_mask.png")
    right_mask_path = os.path.join(output_root, f"{folder_name}_right_mask.png")
    left_objects = get_objects_from_image(left_path, save_path=left_mask_path, mask_only=True)
    right_objects = get_objects_from_image(right_path, save_path=right_mask_path, mask_only=True)
    left_top_dict = build_top_dict(left_objects)
    right_top_dict = build_top_dict(right_objects)
    # draw_top_labels_on_image(left_mask_path, left_mask_path, left_top_dict, all_objects=left_objects)
    # draw_top_labels_on_image(right_mask_path, right_mask_path, right_top_dict, all_objects=right_objects)
    merged = merge_dicts(left_top_dict, right_top_dict)
    has_palet = any(obj["class"] == "palet" for obj in left_objects + right_objects)
    return merged, has_palet

if __name__ == "__main__":

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["directory", "laptop", "tablet", "group_box", "pallet"])

        for folder in os.listdir(image_root):
            folder_path = os.path.join(image_root, folder)
            if not os.path.isdir(folder_path):
                continue

            print(f"\nОбработка палеты: {folder}")
            merged_dict, has_palet = process_pallet_folder(folder_path, folder)
            total_counts = defaultdict(int)

            if has_palet:
                for key, count in merged_dict.items():
                    class_prefix = key.rsplit("_", 2)[0].replace("_top", "")
                    total_counts[class_prefix] += count["count"]
                    print(f"{key}: {count['count']} коробок")


            print(f"\nИтого по классам:")
            for cls in ["laptop", "tablet", "group_box"]:
                print(f"{cls}: {total_counts[cls]} коробок")

            print(f"\nПалета найдена: {'YES' if has_palet else 'NO'}")

            writer.writerow([
                folder,
                0 if not has_palet else total_counts["laptop"],
                0 if not has_palet else total_counts["tablet"],
                0 if not has_palet else total_counts["group_box"],
                "YES" if has_palet else "NO"
            ])


