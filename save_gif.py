import glob
from PIL import Image

# filepaths
fp_out = "toy_data.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(str(f) + ".png") for f in range(40)] + [
    Image.open(str(39) + ".png") for f in range(5)
]
img.save(
    fp=fp_out, format="GIF", append_images=imgs, save_all=True, duration=200, loop=0
)
