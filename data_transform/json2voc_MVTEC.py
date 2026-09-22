import os
import cv2
import math
import numpy as np
import json
import shutil

import xml.etree.ElementTree as ET


# ============================================================
# Path configuration
# ============================================================

BASE_DIR = "/workspace/datasets/MVTec_screws"

JSON_DIR = BASE_DIR

VOC_DIR = os.path.join(BASE_DIR, "VOC")
ANNOTATION_DIR = os.path.join(VOC_DIR, "Annotations")
IMAGESET_DIR = os.path.join(VOC_DIR, "ImageSets")
IMAGE_DIR = os.path.join(VOC_DIR, "JPEGImages")

VOC_CLASSES = [
    "type_001",
    "type_002",
    "type_003",
    "type_004",
    "type_005",
    "type_006",
    "type_007",
    "type_008",
    "type_009",
    "type_010",
    "type_011",
    "type_012",
    "type_013",
]


# ============================================================
# bbox conversion
# ============================================================

def bbox2VOC(bbox):
    """
    Input:
        bbox = [x, y, w, h, theta]

    Return:
        [x, y, w, h, theta]
    """
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    theta = bbox[4]

    return [x, y, w, h, theta]


# ============================================================
# Angle transformation
# ============================================================

def _angle_transform(theta):
    """
    Transform angle into quadrant information.

    This function is retained from the original implementation.
    """

    theta = theta % (2 * math.pi)

    if theta >= 0.0 and theta < (0.5 * math.pi):
        return [1, 1]

    elif theta >= (0.5 * math.pi) and theta < math.pi:
        return [0, 1]

    elif theta >= math.pi and theta < (1.5 * math.pi):
        return [0, 0]

    elif theta >= (1.5 * math.pi):
        return [1, 0]


# ============================================================
# Box conversion class
# ============================================================

class box2VOC():

    def __init__(self, boxes):
        self.boxes = boxes

    def __call__(self, type_):
        """
        Input:
            list with:
            [y, x, h, w, theta, label]

        For txt:
            output rotated 4-point box

        For xml:
            output:
            [x, y, w, h, theta, label]
        """

        boxes_list = []

        for box in self.boxes:

            x = box[1]
            y = box[0]
            w = box[2]
            h = box[3]
            theta = box[4]
            _label = box[5]

            if type_ == "txt":

                box_points = cv2.boxPoints(
                    (
                        (x, y),
                        (w, h),
                        -math.degrees(theta)
                    )
                )

                boxes_list.append(
                    (np.reshape(box_points, 8), _label)
                )

            elif type_ == "xml":

                boxes_list.append(
                    [x, y, w, h, theta, _label]
                )

        if type_ == "xml" and len(boxes_list) > 0:
            boxes_list = np.array(
                boxes_list,
                dtype=np.float64
            )

        return boxes_list

    def _PointsToDataset(self, Points, theta, _label):

        xc = round(
            (Points[0][0] + Points[2][0]) / 2,
            3
        )

        yc = round(
            (Points[0][1] + Points[2][1]) / 2,
            3
        )

        bw = round(
            np.max(Points, axis=0)[0]
            - np.min(Points, axis=0)[0],
            3
        )

        bh = round(
            np.max(Points, axis=0)[1]
            - np.min(Points, axis=0)[1],
            3
        )

        R1 = (
            Points[Points.argmin(axis=0)[1]][0]
            - np.min(Points, axis=0)[0]
        ) / bw

        R1 = round(R1, 5)

        R2 = (
            Points[Points.argmin(axis=0)[0]][1]
            - np.min(Points, axis=0)[1]
        ) / bh

        R2 = round(R2, 5)

        jud, adv = self._angle_transfrom(theta)

        return [xc, yc, bw, bh, _label]


    def _angle_transfrom(self, theta):

        if theta > math.pi:
            theta = theta - 2 * math.pi

        if theta <= 0 and theta > (-0.5 * math.pi):
            return [1, 1]

        elif theta <= (-0.5 * math.pi) and theta > (-math.pi):
            return [0, 1]

        elif (
            theta > (0.5 * math.pi)
            and theta <= math.pi
        ) or theta == -3.141593:

            return [0, 0]

        elif theta > 0 and theta <= (0.5 * math.pi):
            return [1, 0]


# ============================================================
# JSON -> TXT / XML
# ============================================================

def to_yolo_txt(type_i, type_o, dir_name, set_path):

    """
    Convert MVTec JSON annotation.

    type_i:
        json

    type_o:
        txt or xml

    dir_name:
        output annotation directory

    set_path:
        input JSON file
    """

    assert set_path[-4:] == type_i, (
        "Type of file should be *.{} , "
        "file name is {}".format(
            type_i,
            set_path
        )
    )

    ann_path = set_path

    with open(ann_path, "r") as json_file:
        data_tree = json.load(json_file)

    text_file_ = dir_name

    os.makedirs(
        text_file_,
        exist_ok=True
    )

    # ========================================================
    # JSON -> TXT
    # ========================================================

    if type_o == "txt":

        images = data_tree["images"]

        for img_info in images:

            bb = []

            txt_path = open(
                os.path.join(
                    text_file_,
                    img_info["file_name"][0:-4] + ".txt"
                ),
                "w+"
            )

            for ann in data_tree["annotations"]:

                if ann["image_id"] == img_info["id"]:

                    bbox = list(ann["bbox"])

                    bbox.append(
                        VOC_CLASSES[
                            ann["category_id"] - 1
                        ]
                    )

                    bb.append(bbox)

            tt = box2VOC(bb)(type_o)

            for item in tt:

                box = item[0]
                _label = item[1]

                txt_path.write(
                    "{:.1f} {:.1f} {:.1f} {:.1f} "
                    "{:.1f} {:.1f} {:.1f} {:.1f} {} {}\n"
                    .format(
                        box[0],
                        box[1],
                        box[2],
                        box[3],
                        box[4],
                        box[5],
                        box[6],
                        box[7],
                        _label,
                        int(0)
                    )
                )

            txt_path.close()

    # ========================================================
    # JSON -> XML
    # ========================================================

    elif type_o == "xml":

        images = data_tree["images"]
        cats = data_tree["categories"]

        cat_l = [
            {
                "id": i["id"],
                "name": i["name"]
            }
            for i in cats
        ]

        img_list = []

        # ----------------------------------------------------
        # Create XML for every image
        # ----------------------------------------------------

        for img_inf in images:

            root = ET.Element("annotation")

            filename = ET.SubElement(
                root,
                "filename"
            )

            filename.text = img_inf["file_name"]

            size = ET.SubElement(
                root,
                "size"
            )

            width = ET.SubElement(
                size,
                "width"
            )

            width.text = str(
                img_inf["width"]
            )

            height = ET.SubElement(
                size,
                "height"
            )

            height.text = str(
                img_inf["height"]
            )

            depth = ET.SubElement(
                size,
                "depth"
            )

            depth.text = str(3)

            # ------------------------------------------------
            # Add annotations
            # ------------------------------------------------

            for ann in data_tree["annotations"]:

                if ann["image_id"] != img_inf["id"]:
                    continue

                obj = ET.SubElement(
                    root,
                    "object"
                )

                name = ET.SubElement(
                    obj,
                    "name"
                )

                name.text = cat_l[
                    ann["category_id"] - 1
                ]["name"]

                difficult = ET.SubElement(
                    obj,
                    "difficult"
                )

                difficult.text = "0"

                bndbox = ET.SubElement(
                    obj,
                    "bndbox"
                )

                bbox = ann["bbox"]

                xxyyR = bbox2VOC(bbox)

                # --------------------------------------------
                # Original UVC-YOLOX XML format
                # --------------------------------------------

                cx = ET.SubElement(
                    bndbox,
                    "cx"
                )

                cx.text = str(
                    int(xxyyR[1])
                )

                cy = ET.SubElement(
                    bndbox,
                    "cy"
                )

                cy.text = str(
                    int(xxyyR[0])
                )

                w = ET.SubElement(
                    bndbox,
                    "w"
                )

                w.text = str(
                    int(xxyyR[3])
                )

                h = ET.SubElement(
                    bndbox,
                    "h"
                )

                h.text = str(
                    int(xxyyR[2])
                )

                # --------------------------------------------
                # Angle encoding
                # Same as original UVC-YOLOX
                # --------------------------------------------

                a = ET.SubElement(
                    bndbox,
                    "angle"
                )

                angle = (
                    0.8
                    * (
                        (
                            (
                                -xxyyR[4]
                                + 0.5 * math.pi
                            )
                            % (2 * math.pi)
                        )
                        / (2 * math.pi)
                    )
                    + 0.1
                )

                a.text = str(
                    round(
                        float(angle),
                        4
                    )
                )

            # ------------------------------------------------
            # Save XML
            # ------------------------------------------------

            xml_path = os.path.join(
                text_file_,
                img_inf["file_name"][0:-4] + ".xml"
            )

            tree = ET.ElementTree(root)

            tree.write(
                xml_path,
                encoding="utf-8"
            )

            img_list.append(
                img_inf["file_name"][0:-4]
            )

        # ----------------------------------------------------
        # Return image names
        # ----------------------------------------------------

        return img_list


# ============================================================
# Dataset statistics
# ============================================================

def num_class(set_path):

    """
    Calculate MVTec screw class statistics.
    """

    train_path = os.path.join(
        set_path,
        "mvtec_screws_train.json"
    )

    test_path = os.path.join(
        set_path,
        "mvtec_screws_test.json"
    )

    val_path = os.path.join(
        set_path,
        "mvtec_screws_val.json"
    )

    with open(train_path, "r") as train_file:
        train_tree = json.load(train_file)

    with open(test_path, "r") as test_file:
        test_tree = json.load(test_file)

    with open(val_path, "r") as val_file:
        val_tree = json.load(val_file)

    Num_test = np.zeros(
        13,
        dtype=np.uint32
    )

    Num_train = np.zeros(
        13,
        dtype=np.uint32
    )

    for ann in test_tree["annotations"]:

        Num_test[
            ann["category_id"] - 1
        ] += 1

    for ann in train_tree["annotations"]:

        Num_train[
            ann["category_id"] - 1
        ] += 1

    for ann in val_tree["annotations"]:

        Num_train[
            ann["category_id"] - 1
        ] += 1

    return (
        VOC_CLASSES,
        Num_train,
        Num_test
    )


# ============================================================
# Main conversion
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MVTec Screws -> UVC-YOLOX VOC conversion")
    print("=" * 60)

    print("BASE_DIR:")
    print(BASE_DIR)

    print("VOC_DIR:")
    print(VOC_DIR)

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    os.makedirs(
        ANNOTATION_DIR,
        exist_ok=True
    )

    os.makedirs(
        IMAGESET_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Check image directory
    # --------------------------------------------------------

    source_image_dir = os.path.join(
        BASE_DIR,
        "images"
    )

    if not os.path.isdir(source_image_dir):

        raise FileNotFoundError(
            "Image directory not found:\n{}".format(
                source_image_dir
            )
        )

    # --------------------------------------------------------
    # Create JPEGImages symlink
    # --------------------------------------------------------

    if os.path.lexists(IMAGE_DIR):

        if os.path.islink(IMAGE_DIR):

            os.unlink(IMAGE_DIR)

        elif os.path.isdir(IMAGE_DIR):

            shutil.rmtree(IMAGE_DIR)

        else:

            os.remove(IMAGE_DIR)

    os.symlink(
        source_image_dir,
        IMAGE_DIR
    )

    print("JPEGImages -> {}".format(
        source_image_dir
    ))

    # --------------------------------------------------------
    # Convert train / val / test
    # --------------------------------------------------------

    split_names = [
        "train",
        "val",
        "test"
    ]

    for split in split_names:

        json_path = os.path.join(
            JSON_DIR,
            "mvtec_screws_{}.json".format(split)
        )

        if not os.path.isfile(json_path):

            raise FileNotFoundError(
                "JSON file not found:\n{}".format(
                    json_path
                )
            )

        print()
        print("-" * 60)
        print("Converting {}".format(split))
        print("Input:")
        print(json_path)

        img_list = to_yolo_txt(
            type_i="json",
            type_o="xml",
            dir_name=ANNOTATION_DIR,
            set_path=json_path
        )

        # ----------------------------------------------------
        # Create ImageSets/{split}.txt
        # ----------------------------------------------------

        split_txt = os.path.join(
            IMAGESET_DIR,
            "{}.txt".format(split)
        )

        with open(
            split_txt,
            "w"
        ) as f:

            for image_name in img_list:

                f.write(
                    "{}\n".format(image_name)
                )

        print(
            "Images: {}".format(
                len(img_list)
            )
        )

        print(
            "Split file: {}".format(
                split_txt
            )
        )

    # --------------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------------

    clss, train_data, test_data = num_class(
        BASE_DIR
    )

    print()
    print("=" * 60)
    print("Dataset statistics")
    print("=" * 60)

    print("Classes:")
    print(clss)

    print("Train + Val:")
    print(train_data.tolist())

    print(
        "Train + Val total:",
        train_data.sum()
    )

    print("Test:")
    print(test_data.tolist())

    print(
        "Test total:",
        test_data.sum()
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Final verification")
    print("=" * 60)

    for split in split_names:

        split_txt = os.path.join(
            IMAGESET_DIR,
            "{}.txt".format(split)
        )

        with open(split_txt, "r") as f:
            count = len(
                [
                    line
                    for line in f
                    if line.strip()
                ]
            )

        print(
            "{} images: {}".format(
                split,
                count
            )
        )

    xml_files = [
        f
        for f in os.listdir(ANNOTATION_DIR)
        if f.endswith(".xml")
    ]

    print(
        "XML annotations:",
        len(xml_files)
    )

    print()
    print("Conversion completed successfully.")
    print()
    print("VOC dataset:")
    print(VOC_DIR)
