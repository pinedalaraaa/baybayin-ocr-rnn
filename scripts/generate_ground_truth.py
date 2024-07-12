# Script to rename all the image files
import shutil
import os

dir = './images/ya'
renamed = './renamed/ya'

images = [f for f in os.listdir(dir) if f.endswith(('.jpg'))]
print(f"{len(images)} number of images found")

lang = "bybyn"
character = "ya"
part1 = f"{lang}.{character}"

for i, image in enumerate(images):
    # filename = f"{part1}{i}.{image[-3:]}"
    # filename = f"{part1}_{i}.exp.{image[-3:]}"
    filename = f"{part1}_{i}.{image[-3:]}"
    # print(filename)
    shutil.copy(os.path.join(dir, image), os.path.join(renamed, filename))

    # ground truth
    f = open(f"{renamed}/{part1}_{i}.gt.txt", "w")
    f.write(f"ᜌ")
    f.close()