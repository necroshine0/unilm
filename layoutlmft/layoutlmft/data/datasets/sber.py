# coding=utf-8

import json
import os

import datasets
from layoutlmft.data.utils import load_image, normalize_bbox


logger = datasets.logging.get_logger(__name__)


class SberConfig(datasets.BuilderConfig):
    """BuilderConfig for Sber"""

    def __init__(self, **kwargs):
        """BuilderConfig for Sber.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SberConfig, self).__init__(**kwargs)

class Sber(datasets.GeneratorBasedBuilder):
    config = SberConfig(name="sber", version=datasets.Version("1.0.0"), description="Sber dataset")

    tags_names = [
        'type: focus',
        'type: label',
        'type: list, flavour: bul_list',
        'type: list, flavour: enum_list',
        'type: pic',
        'type: pic, flavour: icon',
        'type: plot',
        'type: subtitle',
        'type: table, flavour: mesh',
        'type: table, flavour: mesh, subelement: cell',
        'type: table, flavour: regular_table',
        'type: text',
        'type: timeline',
        'type: title',
        'type: flowchart',
        'type: footnote',
        'type: citation',
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(datasets.features.ClassLabel( names=self.tags_names)),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": "data/sber/train_jsons/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": "data/sber/test_jsons/"}
            ),
        ]

    def _generate_examples(self, filepath):
        with open("data/sber/use_text.txt", "r") as f:
            use_text = (f.read() == True)

        logger.info("⏳ Generating examples from = %s", filepath)
        for guid, file in enumerate(sorted(os.listdir(filepath))):
            tokens, bboxes, ner_tags = [], [], []

            file_path = os.path.join(filepath, file)
            data = json.load(open(file_path, "r", encoding="utf8"))
            image_path = os.path.join("data", "sber", data["meta"]["image"])
            image, size = load_image(image_path)  # resize to 224x224
            for item in data["form"]:
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue

                for w in words:
                    tokens.append(w["text"] if use_text else "")
                    ner_tags.append(label)
                    bboxes.append(normalize_bbox(w["box"], size))

            yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes,
                         "ner_tags": ner_tags, "image": image, "image_path": image_path}
