# Script to rename all the image files
# since training tesseract requires .png files

import shutil
import os

dir = './'
renamed = './'

images = [f for f in os.listdir(dir) if f.endswith(('.jpg'))]
print(f"{len(images)} number of images found")

for i, image in enumerate(images):
    filename = f"{image[:-4]}.png"
    shutil.move(os.path.join(dir, image), os.path.join(renamed, filename))
