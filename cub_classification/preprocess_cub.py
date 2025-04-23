#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from PIL import Image

# preprocessing first, and keeping a copy is more efficient for our uses
# it is possible to preprocess before running a specific image throught he model
# trade-off - efficient to preprocess once vs. anticipating different configurations and write dataloader script to do this as requested


def load_dict_from_txt(txt_path, has_index=True):
    output = {}
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if has_index:
                # first part is an integer ID
                idx = int(parts[0])
                remainder = parts[1:]
                output[idx] = remainder
            else:
                # classes.txt is typically 1 <class_name>, but we want a dict too
                idx = int(parts[0])
                class_name = " ".join(
                    parts[1:]
                )  # in CUB, class_name can have underscores
                output[idx] = class_name
    return output

## resizing because NNs handle fixed-size inputs better
def resize_image_and_bbox(in_path, out_path, bbox, target_size=(224, 224)):
    with Image.open(in_path) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        new_w, new_h = target_size

        # Resize image
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Calculate bbox scale factors
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        x_min, y_min, bbox_w, bbox_h = bbox
        x_min_resized = x_min * scale_x
        y_min_resized = y_min * scale_y
        w_resized = bbox_w * scale_x
        h_resized = bbox_h * scale_y

        x_max_resized = x_min_resized + w_resized
        y_max_resized = y_min_resized + h_resized

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)

    return (x_min_resized, y_min_resized, x_max_resized, y_max_resized)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CUB-200-2011 to train/val CSVs."
    )
    parser.add_argument(
        "--cub_dir",
        type=str,
        default="data/raw/CUB_200_2011",
        help="Path to the CUB-200-2011 folder.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed_cub",
        help="Where to save resized images and CSVs.",
    )
    parser.add_argument(
        "--image_size", type=int, default=224, help="Size for resizing images."
    )
    args = parser.parse_args()

    cub_dir = Path(args.cub_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load mapping from ID -> class_id
    #   e.g., image_class_labels.txt lines: 1 1
    image_class_labels = load_dict_from_txt(cub_dir / "image_class_labels.txt")

    # Load mapping from ID -> train/test (train_test_split.txt lines: 1 1)
    #   1 = train, 0 = test
    train_test_split = load_dict_from_txt(cub_dir / "train_test_split.txt")

    # Load mapping from ID -> bounding box [x,y,w,h]
    #   bounding_boxes.txt lines: 1 60.0 27.0 325.0 304.0
    bounding_boxes = load_dict_from_txt(cub_dir / "bounding_boxes.txt")

    # Load mapping from ID -> relative path
    #   images.txt lines: 1 001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
    images = load_dict_from_txt(cub_dir / "images.txt")

    # Load mapping from class_id -> class_name
    #   classes.txt lines: 1 001.Black_footed_Albatross
    classes = load_dict_from_txt(cub_dir / "classes.txt", has_index=False)

    # We'll do a train.csv and val.csv, mapping test to val for simplicity
    train_csv_path = out_dir / "train.csv"
    val_csv_path = out_dir / "val.csv"

    # Output images directory
    out_images_dir = out_dir / "images"

    # Prepare CSV writers
    train_csv = open(train_csv_path, "w", newline="")
    val_csv = open(val_csv_path, "w", newline="")

    train_writer = csv.writer(train_csv)
    val_writer = csv.writer(val_csv)

    # CSV headers
    headers = ["filename", "class_id", "class_name", "x_min", "y_min", "x_max", "y_max"]
    train_writer.writerow(headers)
    val_writer.writerow(headers)

    # Process each image ID
    for img_id, path_parts in images.items():
        # path_parts should be e.g. ["001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg"]
        rel_path = path_parts[0]
        in_path = cub_dir / "images" / rel_path

        # bounding box
        bbox_parts = bounding_boxes[img_id]  # e.g. ["60.0","27.0","325.0","304.0"]
        x_min = float(bbox_parts[0])
        y_min = float(bbox_parts[1])
        box_w = float(bbox_parts[2])
        box_h = float(bbox_parts[3])

        # class label
        cls_id_str = image_class_labels[img_id][0]  # e.g. "1"
        cls_id = int(cls_id_str)
        cls_name = classes[cls_id]

        # train/test
        # if train_test_split[img_id][0] == "1" => train, else test
        is_train = train_test_split[img_id][0] == "1"

        # Resized output path
        out_path = out_images_dir / rel_path  # replicate subfolders
        # resize + fix bounding box
        x_min_res, y_min_res, x_max_res, y_max_res = resize_image_and_bbox(
            in_path,
            out_path,
            (x_min, y_min, box_w, box_h),
            target_size=(args.image_size, args.image_size),
        )

        row = [
            str(rel_path),  # filename
            cls_id,  # class_id
            cls_name,  # class_name
            f"{x_min_res:.2f}",  # x_min
            f"{y_min_res:.2f}",  # y_min
            f"{x_max_res:.2f}",  # x_max
            f"{y_max_res:.2f}",  # y_max
        ]

        if is_train:
            train_writer.writerow(row)
        else:
            val_writer.writerow(row)

    train_csv.close()
    val_csv.close()

    print("Preprocessing complete!")
    print(f"Train CSV: {train_csv_path}")
    print(f"Val CSV:   {val_csv_path}")
    print(f"Resized images saved under {out_images_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
