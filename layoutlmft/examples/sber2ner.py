import os
import re
import json
import shutil
import argparse
from glob import glob
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_datasets',
        type=str,
        default="data",
    )
    parser.add_argument(
        '--filter_annots',
        type=bool,
        default=False,
    )

    args = parser.parse_args()
    sber_path = os.path.join(args.path_to_datasets, 'sber')
    if not os.path.exists(sber_path):
        raise RuntimeError("sber folder is not exists or invalid path:", sber_path)

    os.makedirs(f"{sber_path}/json", exist_ok=True)
    os.makedirs(f"{sber_path}/train_jsons", exist_ok=True)
    os.makedirs(f"{sber_path}/test_jsons", exist_ok=True)

    with open(f"{sber_path}/result_with_text.json", 'r') as f:
        sber = json.load(f)

    images = sber['images']
    categories = sber['categories']
    annotations = sber['annotations']

    # Build and save annotations in NER format
    for img in images:
        img_id = img['id']
        img_file = img['file_name']
        W, H = img['width'], img['height']
        # Group annotations by image
        img_annots = []
        for ann in annotations:
            if ann['image_id'] == img_id:
                img_annots.append(ann)

        # Filter objects with less than 2 elements
        if args.filter_annots and len(img_annots) <= 1:
            continue

        form = []
        for ann in img_annots:
            text = ann['text']
            category_id = ann['category_id']
            category = categories[ann['category_id']]['name']
            x1, y1, w, h = ann['bbox']

            # Filter invalid bounding boxes
            if x1 + w > W or y1 + h > H:
                continue

            box = [x1, y1, x1 + w, y1 + h]
            cord_like_ann = {
                "label": category,
                "label_id": category_id,
                "words": [{"box": box, "text": text}],
            }

            form.append(cord_like_ann)

        img_json = {
            "form": form,
            "meta": {
                "split": "UNK",
                "image_id": img_id,
                "image": img_file,
                "image_size": {
                    "width": img['width'],
                    "height": img['height']
                }
            },
        }

        # Saving
        file = os.path.split(img_file)[-1].replace('.png', '.json')
        with open(f"{sber_path}/json/{file}", 'w') as f:
            json.dump(img_json, f)

    # Split train/test
    files = glob(f"{sber_path}/json/*")
    print("Dataset size:", len(files))

    files_train, files_test = train_test_split(files, test_size=0.2, random_state=42)
    sets_dict = {"train": files_train, "test": files_test}
    print(f"train/test: {len(files_train)}/{len(files_test)}")

    for set in sets_dict:
        base_files = sets_dict[set]
        for file in base_files:
            new_file = file.replace('/json', f'/{set}_jsons')
            os.rename(file, new_file)

    for set in ['train', 'test']:
        for file in glob(f"{sber_path}/{set}/json/*.json"):
            data = json.load(f, 'r')
            data['meta']['split'] = set
            json.dump(data, open(file, 'w'))
    shutil.rmtree(f"{sber_path}/json")

    # Save images and categories data
    meta = {"images": images, "categories": categories}
    with open(f"{sber_path}/meta.json", 'w', encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

