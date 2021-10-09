import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

from metadata import save_metadata


groups = [
    "bases",
    "backgrounds",
    "wings",
    "bodies",
    "faces",
    "facemarkings",
    "shirts",
    "jewelry",
    "eyewear",
    "hats",
    "hands",
    "foregrounds"
]


body_probabilities = [
    .09, # rainbow
    .0666, # devil
    .1, # clear
    .2, # blue
    .058, # black
    .2, # pink
    .042, # trippy
    .0334, # green glowing
    .01, # rainbow glowing
    .2, # white
]


group_probabilities = {
    "bases": 1,
    "backgrounds": .3,
    "wings": .069,
    "bodies": 1,
    "faces": 1,
    "facemarkings": .3,
    "shirts": .2,
    "jewelry": .15,
    "eyewear": .3,
    "hats": .75,
    "hands": .75,
    "foregrounds": .1
}


blacklist = {
    ("faces", "hats"): [
        (0, 3),
        (0, 11),
        (0, 18),
        (1, 3),
        (1, 17),
        (2, 3),
        (2, 17),
        (2, 18),
        (3, 3),
        (4, 3),
    ],
    ("faces", "facemarkings"): [
        (0, 0),
        (1, 3),
        (1, 8),
        (1, 9),
        (2, 3),
        (2, 8),
        (3, 1),
        (3, 2),
        (3, 4),
        (3, 9),
        (4, 0),
        (3, 10),
        (4, 10),
    ],
    ("eyewear", "hats"): [
        (0, 3),
        (0, 4),
        (0, 6),
        (0, 8),
        (0, 16),
        (1, 17),
        (2, 3),
        (2, 6),
        (2, 16),
        (3, 3),
        (3, 6),
        (3, 16),
        (4, 3),
        (5, 3),
        (5, 6),
        (5, 16),
        (7, 3),
        (7, 6),
        (7, 16),
        (8, 6),
    ],
    ("facemarkings", "eyewear"): [
        (8, 7),
        (8, 8),
        (9, 0),
        (9, 7),
        (9, 8),
        (10, 0),
        (10, 2),
        (10, 3),
        (10, 5),
        (10, 7),
        (10, 8)
    ],
    ("facemarkings", "shirts"): [
        (1, 0),
        (1, 1),
        (4, 0),
        (4, 1),
        (9, 0),
        (0, 0),
    ],
    ("faces", "eyewear"): [
        (0, 2),
        (0, 5),
        (0, 8),
        (2, 0),
        (2, 2),
        (2, 5),
        (2, 8),
    ],
    ("faces", "jewelry"): [
        (3, 3),
    ],
    ("jewerly", "hats"): [
        (0, 3),
        (0, 5),
        (0, 19),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 19),
        (3, 3),
        (3, 19),
    ],
    ("shirts", "hats"): [
        (0, 3),
        (0, 5),
        (0, 17),
        (0, 19),
        (1, 3),
        (1, 5),
        (1, 17),
        (1, 19),
        (9, 2),
    ] + [(i, 17) for i in range(100)],
    ("shirts", "jewelry"): [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 2),
        (0, 3), #tie + pattened shirts
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
        (5, 3),
        (6, 3),
        (8, 3),
        (9, 3),
        (11, 3),
        (0, 2),  #heart + pattened shirts
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),
        (5, 2),
        (6, 2),
        (8, 2),
        (9, 2),
        (11, 2),
        (0, 1), #monake necklace + pattened shirts
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (6, 1),
        (8, 1),
        (9, 1),
        (11, 1),
        # scarves + tanks
        (0, 4),
        (1, 4),
        (4, 4),
        (5, 4),
        (6, 4),
        (7, 4),
        (8, 4),
        (10, 4),
        (0, 5),
        (1, 5),
        (4, 5),
        (5, 5),
        (6, 5),
        (7, 5),
        (8, 5),
        (10, 5),
    ],
    ("wings", "hats"): [
        (1, 1),
        (1, 12),
    ],
    ("facemarkings", "hats"): [
        (0, 3),
        (0, 5),
        (0, 12),
        (0, 17),
        (1, 3),
        (1, 5),
        (3, 5),
        (3, 17),
        (5, 8),
        (5, 10),
        (5, 13),
        (5, 15),
        (5, 19),
        (6, 5),
        (7, 4),
        (7, 16),
        (8, 5),
        (8, 17),
        (9, 5),
        (9, 11),
        (9, 17),
        (3, 14),
        (3, 10),
        (6, 27),
        (6, 39),
        (4, 10),
        (4, 3),
        (4, 39),
        (4, 18),
        (3, 3),
        (4, 40),
        (3, 39),
        (4, 25),
        (4, 30),
        (3, 16),
        (3, 4),
        (4, 19),
        (3, 22),
        (4, 33),
        (4, 16),
        (3, 27),
        (4, 40),
        (3, 34),
        (4, 20),
        (3, 17),
        (4, 17),
        (1, 29),
        (3, 36),
        (3, 30),
        (4, 14),
        (3, 23),
        (3, 13),
        (3, 15),
        (4, 13),
        (10, 5),
    ],
    ("wings", "bodies"): [

    ],
    ("bodies", "shirts"): [

    ] + [(4, i) for i in range(100)],
    ("bodies", "hats"): [
        (4, 5)
    ],
    ("bodies", "facemarkings"): [
        (4, 9)
    ],
    ("hats", "foregrounds"): [
        (4, 5),
        (8, 5),
        (26, 5),
    ],
    ("shirts", "hands"): [
        (9, 38)
    ]
}


def load_pngs(dir=None):
    if dir == None:
        return list(filter(lambda x: x[-3:] == "png", os.listdir()))
    return list(map(lambda x: dir + "/" + x, filter(lambda x: x[-3:] == "png", os.listdir(dir))))

group_images = {group: load_pngs("features/" + group) for group in groups}


def run_cross_product(first_family, first_lower_bound, second_family, second_lower_bound):
    os.mkdir(f"cross_products_v2/{first_family}_{second_family}")
    for j, first_path in enumerate(load_pngs("features/" + first_family)):
        first_number = int(first_path.split("/")[-1][:-4])
        if first_number >= first_lower_bound:
            for k, second_path in enumerate(load_pngs("features/" + second_family)):
                if int(second_path.split("/")[-1][:-4]) >= second_lower_bound:
                    print(first_path, second_path)
                    base = Image.open("features/bases/White base.png")
                    first = Image.open(first_path)
                    second = Image.open(second_path)
                    base.paste(first, (0, 0), first)
                    base.paste(second, (0, 0), second)
                    base.save(f"cross_products_v2/{first_family}_{second_family}/{first_family}{j}_{second_family}{k}.png", "PNG")

def generate(i):
    base_path = np.random.choice(group_images["bases"])
    base = Image.open(base_path)
    elements = {}
    for group in groups[1:]:
        chosen = np.random.random() < group_probabilities[group]
        if chosen:
            if group == "bodies":
                body_idx = np.random.choice(np.arange(10), p=body_probabilities)
                image_path = f"features/bodies/{body_idx}.png"
            else:
                image_path = np.random.choice(group_images[group])
            image_number = int(image_path.split("/")[-1][:-4])

            # print(group, image_number)
            for first_group, number in elements.items():
                # print(first_group, group)
                if (first_group, group) in blacklist:
                    # print("blacklist record detected", first_group, group, number, image_number)
                    if (number, image_number) in blacklist[(first_group, group)]:
                        # print("failed due to blacklist")
                        return
                if (group, first_group) in blacklist:
                    # print("blacklist record detected REVERSE", group, first_group)
                    if (image_number, number) in blacklist[(group, first_group)]:
                        # print("failed due to blacklist REVERSE")
                        return
            elements[group] = image_number

            image = Image.open(image_path)
            # Inverse bodies need inverese faces
            if group == "faces" and elements["bodies"] == 4:
                # print("Using inverse face!")
                image = Image.open(f"features/inverse_faces/{image_number}.png")
            # Sukuna markings and shirt
            if group == "facemarkings" and image_number == 9 and "shirts" in elements:
                return
            # Sukuna markings and shirt
            if group == "hats" and image_number == 5 and "shirts" in elements:
                return
            if group == "hats" and image_number == 19 and "shirts" in elements:
                return
            if group == "hats" and image_number == 50 and "eyewear" in elements:
                return
            if group == "faces" and image_number == 3 and "shirts" in elements:
                return
            if group == "jewelry" and image_number == 6 and "shirts" in elements:
                return

            if group not in ["backgrounds"]:
                image = center(image)

            base.paste(image, (0, 0), image)

    A = base.getchannel('A')
    base.putalpha(A.point(lambda i: 255))
    base.save(f"replacement_art/{i}.png", "PNG")
    save_metadata(i, [(group, number) for group, number in elements.items()])
    return True


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def center(image):
    margin_image = add_margin(image, 0, 0, 0, 40, (0, 0, 0, 0))
    return margin_image.crop((0, 0, 1000, 1000))

replaces = [
    211,
    448,
    1105,
    1343,
    1512,
    1724,
    1754,
    1806,
    1843,
    1884,
    1915,
    2120,
    2991,
    3064,
    3269,
    3471,
    3605,
    4865,
    5041,
    5081,
    5187,
    5234,
    5284,
    5333,
    5431,
    5548,
    5683,
    5710,
    5793,
    5945,
    6049,
    6114,
    6133,
    6201,
    6220,
    6484,
    6627,
    6630,
    6725,
    6744,
    6823,
    6838,
    6910,
    6921,
    6928,
    7004,
    7080,
    7085,
    7133,
    7166,
    7176,
    7223,
    7264,
    7315,
    7384,
    7515,
    7525,
    7570,
    7666,
    7771,
    7887,
    8017,
    8022,
    8141,
    8191,
    8263,
    8278,
    8363,
    8373,
    8474,
    8521,
    8557,
    8588,
    8901,
    8985,
    9105,
    9205,
    9209,
]


# if not os.path.exists("replacement_art"):
#     os.mkdir("replacement_art")
# if not os.path.exists("replacement_metadata"):
#     os.mkdir("replacement_metadata")
# for i in replaces:
#     print("Generating face", i, "...")
#     while not generate(i):
#         continue


def generate_specific(items):
    base_path = f"features/bases/{items[0][1]}.png"
    base = Image.open(base_path)
    for group, number in items[1:]:
        image_path = f"features/{group}/{number}.png"
        image = Image.open(image_path)
        if group not in ["backgrounds"]:
            image = center(image)
        base.paste(image, (0, 0), image)
    A = base.getchannel('A')
    base.putalpha(A.point(lambda i: 255))
    base.save("sample.png", "PNG")


generate_specific([
    ("bases", 7),
    ("wings", 0),
    ("bodies", 8),
    ("faces", 1),
    ("facemarkings", 6),
    ("hats", 26),
    ("hands", 8),
])


# for i in range(len(groups))[:-1]:
#     for j in range(len(groups))[i + 1:]:
#         first_group = groups[i]
#         second_group = groups[j]
#         if (first_group, second_group) in processed_groups:
        # first_group =
        # run_cross_product("faces", 0, "hats", 20)
