import numpy as np
from PIL import ImageColor

# COLOR_PALETTE_HEX = ["#581845", "#900c3f", "#c70039", "#ff5733", "#ffc300"]
# COLOR_PALETTE_HEX = ["#173f5f", "#20639b", "#3caea3", "#f6d55c", "#ed553b", "#ffc300"]
# COLOR_PALETTE_HEX = ["#003f5c", "#444e86", "#955196", "#dd5182", "#ff6e54", "#ffa600"]
COLOR_PALETTE_HEX = ["#fc2847", "#ffa343", "#fdfc74", "#71bc78", "#0f4c81", "#7442c8", "#fb7efd"]
COLOR_PALETTE = [np.array(ImageColor.getrgb(color)) / 255 for color in COLOR_PALETTE_HEX]
ALGORITHMS = ["RSPO", "DIPG", "VPG", "DvD", "RND", "RPG"]
# COLOR_MAP = {
#     "RSPO": 4,
#     "DIPG": 3,
#     "VPG": 0,
#     "RND": 1,
#     "RPG": 5,
#     "MAVEN": 2
# }
COLOR_MAP = {
    "RSPO": 0,
    "DIPG": 6,
    "VPG": 4,
    "RND": 1,
    "RPG": 3,
    "MAVEN": 2
}


def get_palette(algs):
    palette = []
    for alg in algs:
        palette.append(COLOR_PALETTE[COLOR_MAP[alg]])
    return palette