"""METRICA semantic taxonomy and mappings from model-specific label sets.

The taxonomy is deliberately extensible: add a category to ``CATEGORIES`` and map model
labels to it. Reconstruction strategies are chosen per *category*, not per model label.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    key: str
    group: str        # agent | vehicle | structure | ground | furniture | vegetation | sky | object
    display: str
    dynamic: bool     # can move (gets tracked); static categories become scene objects
    color: str        # UI colour (hex)


CATEGORIES: dict[str, Category] = {c.key: c for c in [
    Category("person", "agent", "Person", True, "#e4572e"),
    Category("bicycle", "vehicle", "Bicycle", True, "#f3a712"),
    Category("motorcycle", "vehicle", "Motorcycle", True, "#f3a712"),
    Category("car", "vehicle", "Car", True, "#2e86ab"),
    Category("truck", "vehicle", "Truck", True, "#1b4965"),
    Category("bus", "vehicle", "Bus", True, "#5fa8d3"),
    Category("train", "vehicle", "Train", True, "#62b6cb"),
    Category("boat", "vehicle", "Boat", True, "#62b6cb"),
    Category("airplane", "vehicle", "Aircraft", True, "#62b6cb"),
    Category("animal", "agent", "Animal", True, "#a1c181"),
    Category("road", "ground", "Road", False, "#6c6f7d"),
    Category("sidewalk", "ground", "Sidewalk", False, "#b8b8c8"),
    Category("terrain", "ground", "Terrain", False, "#9dbf9e"),
    Category("ground", "ground", "Ground (unclassified)", False, "#8d8d8d"),
    Category("building", "structure", "Building", False, "#c9ada7"),
    Category("wall", "structure", "Wall", False, "#a39594"),
    Category("facade", "structure", "Facade plane", False, "#d4b483"),
    Category("fence", "structure", "Fence", False, "#9a8c98"),
    Category("pole", "structure", "Pole / post", False, "#f2e94e"),
    Category("traffic_light", "structure", "Traffic light", False, "#ffbe0b"),
    Category("traffic_sign", "structure", "Traffic sign", False, "#fb5607"),
    Category("vegetation", "vegetation", "Vegetation", False, "#3a7d44"),
    Category("sky", "sky", "Sky", False, "#87ceeb"),
    Category("bench", "furniture", "Bench", False, "#8d6346"),
    Category("fire_hydrant", "structure", "Fire hydrant", False, "#d62828"),
    Category("parking_meter", "structure", "Parking meter", False, "#6d6875"),
    Category("chair", "furniture", "Chair", False, "#8d6346"),
    Category("couch", "furniture", "Couch", False, "#8d6346"),
    Category("table", "furniture", "Table", False, "#8d6346"),
    Category("bed", "furniture", "Bed", False, "#8d6346"),
    Category("tv", "object", "Screen / TV", False, "#457b9d"),
    Category("bag", "object", "Bag", True, "#b5838d"),
    Category("suitcase", "object", "Suitcase", True, "#b5838d"),
    Category("bottle", "object", "Bottle", True, "#b5838d"),
    Category("potted_plant", "vegetation", "Potted plant", False, "#52b788"),
    Category("object", "object", "Object (other)", False, "#adb5bd"),
    Category("unknown", "object", "Unclassified geometry", False, "#ced4da"),
]}

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

# COCO label -> METRICA category. Labels not listed map to "object" (kept, not discarded).
COCO_TO_CATEGORY = {
    "person": "person", "bicycle": "bicycle", "car": "car", "motorcycle": "motorcycle", "bus": "bus",
    "train": "train", "truck": "truck", "boat": "boat", "airplane": "airplane",
    "traffic light": "traffic_light", "stop sign": "traffic_sign", "fire hydrant": "fire_hydrant",
    "parking meter": "parking_meter", "bench": "bench", "chair": "chair", "couch": "couch",
    "dining table": "table", "bed": "bed", "tv": "tv", "backpack": "bag", "handbag": "bag",
    "suitcase": "suitcase", "bottle": "bottle", "potted plant": "potted_plant",
    "bird": "animal", "cat": "animal", "dog": "animal", "horse": "animal", "sheep": "animal", "cow": "animal",
}

# Detected classes that are tracked over time by default (others are kept as static evidence).
TRACKED_CATEGORIES = {"person", "bicycle", "motorcycle", "car", "truck", "bus", "train", "animal", "boat",
                      "bag", "suitcase"}

CITYSCAPES_NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
                    "motorcycle", "bicycle"]
CITYSCAPES_TO_CATEGORY = {
    "road": "road", "sidewalk": "sidewalk", "building": "building", "wall": "wall", "fence": "fence",
    "pole": "pole", "traffic light": "traffic_light", "traffic sign": "traffic_sign",
    "vegetation": "vegetation", "terrain": "terrain", "sky": "sky", "person": "person", "rider": "person",
    "car": "car", "truck": "truck", "bus": "bus", "train": "train", "motorcycle": "motorcycle",
    "bicycle": "bicycle",
}

# Dense scene-label map used internally (uint8 ids). 0 = unknown.
SCENE_LABELS = ["unknown", "road", "sidewalk", "terrain", "building", "wall", "fence", "pole", "traffic_light",
                "traffic_sign", "vegetation", "sky", "person", "car", "truck", "bus", "train", "motorcycle",
                "bicycle", "ground"]
SCENE_ID = {k: i for i, k in enumerate(SCENE_LABELS)}
GROUND_IDS = {SCENE_ID[k] for k in ("road", "sidewalk", "terrain", "ground")}
DYNAMIC_IDS = {SCENE_ID[k] for k in ("person", "car", "truck", "bus", "train", "motorcycle", "bicycle")}
STRUCTURE_IDS = {SCENE_ID[k] for k in ("building", "wall", "fence")}


def category_for_coco(name: str) -> str:
    return COCO_TO_CATEGORY.get(name, "object")


def category_group(cat: str) -> str:
    return CATEGORIES.get(cat, CATEGORIES["object"]).group
