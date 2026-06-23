import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
DEFAULT_CLASS_NAMES = {
    0: "crack",
    1: "dent",
    2: "glass shatter",
    3: "lamp broken",
    4: "scratch",
    5: "tire flat",
}


def calculate_md5(file_path):
    h = hashlib.md5()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_group_key(filename):
    base = os.path.basename(filename).lower()
    if base.endswith(IMAGE_EXTENSIONS) or base.endswith(".txt"):
        stem = os.path.splitext(base)[0]
    else:
        stem = base
    if ".rf." in stem:
        stem = stem.split(".rf.", 1)[0]
    if "~v" in stem:
        stem = stem.split("~v", 1)[0]
    return stem


def find_image_for_stem(images_dir, stem):
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(images_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def read_label_classes(label_path):
    class_counts = Counter()
    invalid_lines = []
    with open(label_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
                class_counts[class_id] += 1
            except Exception:
                invalid_lines.append({"line_no": line_no, "line": line.strip()})
    return class_counts, invalid_lines


def bbox_line_to_rectangle_segment(line):
    parts = line.strip().split()
    if len(parts) != 5:
        return line.rstrip("\n")
    class_id, x_center, y_center, width, height = parts
    x_center = float(x_center)
    y_center = float(y_center)
    width = float(width)
    height = float(height)
    x1 = max(0.0, min(1.0, x_center - width / 2))
    y1 = max(0.0, min(1.0, y_center - height / 2))
    x2 = max(0.0, min(1.0, x_center + width / 2))
    y2 = max(0.0, min(1.0, y_center + height / 2))
    return f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"


def copy_label_as_rectangle_segments(src_label, dst_label):
    os.makedirs(os.path.dirname(dst_label), exist_ok=True)
    converted_lines = []
    with open(src_label, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            converted_lines.append(bbox_line_to_rectangle_segment(stripped))
    with open(dst_label, "w", encoding="utf-8", newline="\n") as handle:
        if converted_lines:
            handle.write("\n".join(converted_lines) + "\n")


def _find(parent, key):
    parent.setdefault(key, key)
    if parent[key] != key:
        parent[key] = _find(parent, parent[key])
    return parent[key]


def _union(parent, left, right):
    root_left = _find(parent, left)
    root_right = _find(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left


def collect_annotated_items(project_root, class_names=None):
    class_names = class_names or DEFAULT_CLASS_NAMES
    data7_dir = os.path.join(project_root, "Data7.off")
    images_dir = os.path.join(data7_dir, "images")
    labels_dir = os.path.join(data7_dir, "labels")
    items = []
    issues = {
        "missing_images": [],
        "invalid_label_lines": [],
        "nonstandard_classes": [],
    }

    if not os.path.isdir(labels_dir):
        return items, issues

    parent = {}
    hash_to_key = {}
    raw_items = []

    for label_name in sorted(f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")):
        stem = os.path.splitext(label_name)[0]
        label_path = os.path.join(labels_dir, label_name)
        image_path = find_image_for_stem(images_dir, stem)
        if not image_path:
            issues["missing_images"].append(label_name)
            continue

        class_counts, invalid_lines = read_label_classes(label_path)
        if invalid_lines:
            issues["invalid_label_lines"].append({"file": label_name, "lines": invalid_lines})

        invalid_classes = [cid for cid in class_counts if cid not in class_names]
        if invalid_classes:
            issues["nonstandard_classes"].append({"file": label_name, "class_ids": invalid_classes})

        source_key = source_group_key(stem)
        _find(parent, source_key)
        image_hash = calculate_md5(image_path)
        if image_hash in hash_to_key:
            _union(parent, source_key, hash_to_key[image_hash])
        else:
            hash_to_key[image_hash] = source_key

        raw_items.append({
            "stem": stem,
            "image_name": os.path.basename(image_path),
            "label_name": label_name,
            "image_path": image_path,
            "label_path": label_path,
            "source_key": source_key,
            "image_hash": image_hash,
            "class_counts": dict(class_counts),
        })

    for item in raw_items:
        item["group_key"] = _find(parent, item["source_key"])
        items.append(item)

    return items, issues


def _stable_order(keys, seed):
    return sorted(keys, key=lambda key: hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest())


def build_split_assignments(items, val_ratio=0.15, test_ratio=0.15, seed=42, locked_test_group_keys=None):
    locked_test_group_keys = set(locked_test_group_keys or [])
    groups = {}
    for item in items:
        key = item["group_key"]
        if key not in groups:
            groups[key] = {"items": [], "class_counts": Counter()}
        groups[key]["items"].append(item)
        groups[key]["class_counts"].update({int(k): int(v) for k, v in item["class_counts"].items()})

    all_group_keys = set(groups.keys())
    split_groups = {"train": set(), "val": set(), "test": set()}
    split_groups["test"] = locked_test_group_keys & all_group_keys

    total_images = len(items)
    target_test_images = max(1, round(total_images * test_ratio)) if total_images >= 5 else 0
    target_val_images = max(1, round(total_images * val_ratio)) if total_images >= 5 else 0

    class_group_totals = Counter()
    for key, group in groups.items():
        for class_id in group["class_counts"]:
            class_group_totals[class_id] += 1

    heldout_class_groups = Counter()
    for key in split_groups["test"]:
        for class_id in groups[key]["class_counts"]:
            heldout_class_groups[class_id] += 1

    def group_image_count(key):
        return len(groups[key]["items"])

    def split_image_count(split):
        return sum(group_image_count(key) for key in split_groups[split])

    def can_holdout(key):
        if key in split_groups["test"] or key in split_groups["val"]:
            return False
        for class_id in groups[key]["class_counts"]:
            if class_group_totals[class_id] - heldout_class_groups[class_id] <= 1:
                return False
        return True

    def add_group(split, key):
        split_groups[split].add(key)
        for class_id in groups[key]["class_counts"]:
            heldout_class_groups[class_id] += 1

    ordered_keys = _stable_order(list(all_group_keys - split_groups["test"]), seed)
    classes_by_rarity = [cid for cid, _ in sorted(class_group_totals.items(), key=lambda item: (item[1], item[0]))]

    for split, target, min_groups_for_class in [("test", target_test_images, 2), ("val", target_val_images, 3)]:
        for class_id in classes_by_rarity:
            if class_group_totals[class_id] < min_groups_for_class:
                continue
            if any(class_id in groups[key]["class_counts"] for key in split_groups[split]):
                continue
            candidates = [key for key in ordered_keys if class_id in groups[key]["class_counts"] and can_holdout(key)]
            if candidates:
                add_group(split, min(candidates, key=lambda key: (group_image_count(key), key)))

        for key in ordered_keys:
            if split_image_count(split) >= target:
                break
            if can_holdout(key):
                add_group(split, key)

    used = split_groups["val"] | split_groups["test"]
    split_groups["train"] = all_group_keys - used

    assignments = {}
    for split, keys in split_groups.items():
        for key in keys:
            for item in groups[key]["items"]:
                assignments[item["stem"]] = split

    return assignments, split_groups


def _clear_split_dir(split_root):
    for cache_name in ["labels.cache", "images.cache"]:
        cache_path = os.path.join(split_root, cache_name)
        if os.path.exists(cache_path):
            os.remove(cache_path)
    for subdir in ["images", "labels"]:
        path = os.path.join(split_root, subdir)
        os.makedirs(path, exist_ok=True)
        for cache_name in ["labels.cache", "images.cache"]:
            cache_path = os.path.join(path, cache_name)
            if os.path.exists(cache_path):
                os.remove(cache_path)
        for name in os.listdir(path):
            full_path = os.path.join(path, name)
            if name == ".gitkeep":
                continue
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)


def summarize_dataset_split(dataset_root):
    summary = {}
    hash_to_location = {}
    duplicate_hashes = []

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(dataset_root, split, "images")
        lbl_dir = os.path.join(dataset_root, split, "labels")
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTENSIONS)] if os.path.isdir(img_dir) else []
        labels = [f for f in os.listdir(lbl_dir) if f.lower().endswith(".txt")] if os.path.isdir(lbl_dir) else []
        class_counts = Counter()
        boxes = 0
        missing_labels = 0
        missing_images = 0

        for image_name in images:
            stem = os.path.splitext(image_name)[0]
            label_path = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.exists(label_path):
                missing_labels += 1
            else:
                counts, _ = read_label_classes(label_path)
                class_counts.update(counts)
                boxes += sum(counts.values())

            image_path = os.path.join(img_dir, image_name)
            image_hash = calculate_md5(image_path)
            if image_hash in hash_to_location and hash_to_location[image_hash]["split"] != split:
                duplicate_hashes.append({
                    "image": image_name,
                    "split": split,
                    "other": hash_to_location[image_hash],
                })
            else:
                hash_to_location[image_hash] = {"split": split, "image": image_name}

        image_stems = {os.path.splitext(name)[0] for name in images}
        for label_name in labels:
            if os.path.splitext(label_name)[0] not in image_stems:
                missing_images += 1

        summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": boxes,
            "class_counts": dict(sorted(class_counts.items())),
            "missing_labels": missing_labels,
            "missing_images": missing_images,
        }

    summary["duplicate_hashes_across_splits"] = duplicate_hashes
    return summary


def write_data_yaml(dataset_root, class_names=None):
    class_names = class_names or DEFAULT_CLASS_NAMES
    yaml_path = os.path.join(dataset_root, "data.yaml")
    root_for_yaml = dataset_root.replace("\\", "/")
    names_yaml = "\n".join(f"  {class_id}: {name}" for class_id, name in sorted(class_names.items()))
    content = f'''path: "{root_for_yaml}"
train: "train/images"
val: "val/images"
test: "test/images"

names:
{names_yaml}
'''
    with open(yaml_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    return yaml_path


def load_locked_test_group_keys(project_root):
    manifest_path = os.path.join(project_root, "dataset_annote", "split_manifest.json")
    if not os.path.exists(manifest_path):
        return set()
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        return {item["group_key"] for item in manifest.get("items", []) if item.get("split") == "test"}
    except Exception:
        return set()


def sync_existing_split_item(project_root, stem):
    manifest_path = os.path.join(project_root, "dataset_annote", "split_manifest.json")
    if not os.path.exists(manifest_path):
        return None

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    match = None
    for item in manifest.get("items", []):
        if item.get("stem") == stem:
            match = item
            break
    if not match:
        return None

    split = match["split"]
    image_name = match["image_name"]
    label_name = match["label_name"]
    src_image = os.path.join(project_root, "Data7.off", "images", image_name)
    src_label = os.path.join(project_root, "Data7.off", "labels", label_name)
    dst_image = os.path.join(project_root, "dataset_annote", split, "images", image_name)
    dst_label = os.path.join(project_root, "dataset_annote", split, "labels", label_name)

    if not os.path.exists(src_image) or not os.path.exists(src_label):
        return None

    os.makedirs(os.path.dirname(dst_image), exist_ok=True)
    os.makedirs(os.path.dirname(dst_label), exist_ok=True)
    shutil.copy2(src_image, dst_image)
    copy_label_as_rectangle_segments(src_label, dst_label)
    return {
        "split": split,
        "image_path": dst_image,
        "label_path": dst_label,
    }


def prepare_train_val_test_split(project_root, val_ratio=0.15, test_ratio=0.15, seed=42, lock_existing_test=True, class_names=None):
    class_names = class_names or DEFAULT_CLASS_NAMES
    dataset_root = os.path.join(project_root, "dataset_annote")
    items, issues = collect_annotated_items(project_root, class_names=class_names)
    if not items:
        raise ValueError("Aucune image annotée exploitable trouvée dans Data7.off.")
    if issues["nonstandard_classes"]:
        raise ValueError("Des classes hors classement best_2.pt sont encore présentes dans Data7.off.")

    locked_test_group_keys = load_locked_test_group_keys(project_root) if lock_existing_test else set()
    assignments, split_groups = build_split_assignments(
        items,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        locked_test_group_keys=locked_test_group_keys,
    )

    for split in ["train", "val", "test"]:
        _clear_split_dir(os.path.join(dataset_root, split))

    for item in items:
        split = assignments[item["stem"]]
        target_img = os.path.join(dataset_root, split, "images", item["image_name"])
        target_lbl = os.path.join(dataset_root, split, "labels", item["label_name"])
        shutil.copy2(item["image_path"], target_img)
        copy_label_as_rectangle_segments(item["label_path"], target_lbl)

    yaml_path = write_data_yaml(dataset_root, class_names=class_names)
    summary = summarize_dataset_split(dataset_root)

    manifest_items = []
    for item in items:
        manifest_items.append({
            "stem": item["stem"],
            "image_name": item["image_name"],
            "label_name": item["label_name"],
            "group_key": item["group_key"],
            "source_key": item["source_key"],
            "image_hash": item["image_hash"],
            "split": assignments[item["stem"]],
            "class_counts": item["class_counts"],
        })

    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Data7.off",
        "output": "dataset_annote",
        "class_names": {str(k): v for k, v in sorted(class_names.items())},
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "lock_existing_test": lock_existing_test,
        "locked_test_groups_count": len(locked_test_group_keys),
        "leak_prevention": [
            "same source filename before .rf. stays in one split",
            "exact duplicate image hashes stay in one split",
            "test split is never referenced by training train/val paths",
        ],
        "summary": summary,
        "issues": issues,
        "label_format": "YOLO segmentation rectangles generated from Data7.off bounding boxes",
        "items": manifest_items,
    }
    manifest_path = os.path.join(dataset_root, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return {
        "dataset_root": dataset_root,
        "yaml_path": yaml_path,
        "manifest_path": manifest_path,
        "summary": summary,
        "issues": issues,
        "split_groups": {split: len(keys) for split, keys in split_groups.items()},
    }


if __name__ == "__main__":
    result = prepare_train_val_test_split(os.getcwd())
    print(json.dumps(result, ensure_ascii=False, indent=2))
