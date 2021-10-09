import os
import numpy as np
import json

HANDS_NAMES = {
    28: "Baseball Bat",
    33: "Tennis Racket",
    3: "Candy Corn",
    11: "Green Onion",
    2: "Scythe",
    16: "Pitchfork",
    13: "Bowling Ball",
    34: "Apple",
    36: "Teddy Bear",
    22: "Cotton Candy",
    42: "Cupcake",
    52: "Skittles",
    40: "Cheese",
    37: "Tote Bag 1",
    10: "Donut",
    25: "Futbol",
    5: "Spider",
    46: "Pork Bun",
    20: "Football",
    18: "Turtle",
    51: "Hi-Chew",
    12: "Baseball",
    6: "Sword",
    0: "Hook",
    35: "Heavy Bag",
    47: "Volleyball",
    48: "Onigiri",
    53: "M&Ms",
    17: "Pineapple",
    24: "Shaker",
    50: "Mushroom",
    38: "Popsicle",
    39: "Balloon",
    21: "Slice",
    7: "Carrot",
    14: "Hammer",
    4: "Boba",
    19: "Headphones",
    32: "Pencil",
    45: "Dango",
    44: "Borgir",
    23: "Thin Watermelon",
    43: "Sucker",
    15: "Tote Bag 2",
    1: "Lollipop",
    8: "Empty Bag",
    26: "Ice Cream",
    30: "Dumbbell",
    29: "Drumstick",
    41: "Fish",
    49: "Bok Choy",
    9: "Basketball",
    31: "Flower",
}


SHIRTS_NAMES = {
    8: "Cheese Tank",
    3: "Orange Panda Shirt",
    6: "Purple Ice Cream Tank",
    11: "Pink Panda Shirt",
    1: "Yellow Squiggle Tank",
    0: "Turtle Tank",
    7: "Blue Tank",
    9: "Trench Coat",
    5: "Gray Squiggle Tank",
    4: "Blue Ice Cream Tank",
    10: "Mauve Tank",
    2: "Mom Shirt",
}


FOREGROUNDS_NAMES = {
    1: "Bones",
    2: "Eyeball",
    0: "Black Cat",
    4: "Witch’s Pot",
    5: "Boo!",
}


JEWELRY_NAMES = {
    0: "Cross",
    6: "Buttons",
    4: "Green Scarf",
    5: "Orange Scarf",
    2: "Heart",
    3: "Tie",
    1: "Necklace",
}


WINGS_NAMES = {
    1: "Devil",
    0: "Angel",
}


EYEWEAR_NAMES = {
    7: "Glasses",
    6: "Eyepatch",
    3: "Yellow Sunglasses",
    4: "Monocle",
    8: "Spiral Glasses",
    2: "Pink Sunglasses",
    5: "Sunglasses",
    0: "Heart Sunglasses",
    1: "Blindfold",
}


BODIES_NAMES = {
    1: "Devil",
    8: "Glowing Rainbow",
    0: "Rainbow",
    6: "Trippy",
    9: "White",
    2: "Transparent",
    5: "Lavender",
    3: "Blue",
    7: "Glow",
    4: "Black",
}


FACES_NAMES = {
    0: "Devious",
    2: "Spooky",
    1: "Happy",
    3: "Silly",
    4: "Angry",
}


HATS_NAMES = {
    49: "Merry Hat",
    31: "Panda Hat",
    4: "Blue ETH Cap",
    7: "Luffy Straw Hat",
    19: "Squid Game Mask",
    1: "Antlers",
    26: "Squiggle Cap",
    20: "Blue Snowfro",
    12: "Devil Horns",
    17: "Pikachu",
    46: "White Snowfro",
    24: "Halo",
    48: "Santa Hat",
    25: "Avatar",
    50: "Purple Ski Mask",
    30: "Lavender Beanie",
    44: "Orange Party Hat",
    37: "Chicken Hat",
    42: "Green Beanie",
    41: "Pig Hat",
    36: "Viking Helm",
    3: "No-Face",
    38: "Hamster Hat",
    5: "Mononoke",
    39: "Penguin Hat",
    15: "Witch Hat",
    28: "Beanie",
    21: "Flower",
    29: "Visor",
    2: "Frog Hat",
    40: "Army Beanie",
    13: "Pirate Hat",
    45: "Jester Hat",
    23: "Rose",
    27: "Sailor Hat",
    0: "Mohawk",
    32: "Sheriff Hat",
    8: "Red ETH Cap",
    16: "Red Snapback",
    51: "Mistletoe",
    35: "Cow Bucket",
    11: "Hello Kitty",
    33: "Top Hat",
    9: "Dino",
    10: "Chopper",
    43: "Graduation Cap",
    22: "Crown",
    6: "Ninja Headband",
    47: "Earmuffs",
    14: "Bow",
}


FACEMARKINGS_NAMES = {
    10: "Carrot Nose",
    5: "Wound",
    9: "Tribal Tattoos",
    3: "Tattoos",
    1: "Mustache",
    0: "Scar",
    7: "Band-aid",
    4: "Goatee",
    6: "Stitch",
}


BACKGROUNDS_NAMES = {
    0: "Spiderweb",
    7: "Snowy Tree",
    5: "Gravestone",
    4: "Owl",
    2: "Moon",
    6: "Snow",
}

GROUP_NAME_MAP = {
    "backgrounds":  ("Background", BACKGROUNDS_NAMES),
    "wings":        ("Wings", WINGS_NAMES),
    "bodies":       ("Body", BODIES_NAMES),
    "faces":        ("Face", FACES_NAMES),
    "facemarkings": ("Face Attribute", FACEMARKINGS_NAMES),
    "shirts":       ("Clothing", SHIRTS_NAMES),
    "jewelry":      ("Accessory", JEWELRY_NAMES),
    "eyewear":      ("Eyewear", EYEWEAR_NAMES),
    "hats":         ("Headgear", HATS_NAMES),
    "hands":        ("Item", HANDS_NAMES),
    "foregrounds":  ("Special", FOREGROUNDS_NAMES),
}


def save_metadata(token_id, traits):
    attributes = []
    for group, number in traits:
        attributes.append({
            "trait_type": GROUP_NAME_MAP[group][0],
            "value": GROUP_NAME_MAP[group][1][number]
        })
    md_json = {
        "name": f"Spookie Squiggle #{token_id}",
        "description": f"Spookie Squiggle #{token_id} of 9,212",
        "attributes": attributes,
        "image": f"https://spookie-squiggle.s3.us-west-1.amazonaws.com/art/{token_id}.png"
    }
    json.dump(md_json, open(f"replacement_metadata/{token_id}", "w"), indent=4)
