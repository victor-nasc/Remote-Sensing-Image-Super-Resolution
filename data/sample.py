import os
import random
from PIL import Image

move_dir = './'
data_path = 'AID' # path to the dataset

for c in os.listdir(data_path):
    class_path = os.path.join(data_path, c)

    images = os.listdir(class_path)
    selected_images = random.sample(images, 100)

    for image in selected_images:
        img = Image.open(os.path.join(class_path, image)).convert('RGB')
        img = img.resize((512, 512), Image.BICUBIC)
        img.save(os.path.join(move_dir, image) + '.jpg')

    print(f'{c} done')
