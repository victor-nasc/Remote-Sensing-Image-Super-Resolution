import os
import random
from PIL import Image

move_dir = './'  # Directory to move images into
data_path = 'AID'  # Path to the dataset

for c in os.listdir(data_path):
    class_path = os.path.join(data_path, c)
    images = os.listdir(class_path)

    split = [100, 10, 10]
    split_labels = ['train', 'val', 'test']

    selected_images = random.sample(images, sum(split))

    start_idx = 0
    for task, count in zip(split_labels, split):
        os.makedirs(os.path.join(move_dir, task), exist_ok=True)
        
        subset = selected_images[start_idx:start_idx + count]
        start_idx += count

        for image in subset:
            img = Image.open(os.path.join(class_path, image)).convert('RGB')
            img = img.resize((512, 512), Image.BICUBIC)
            img.save(os.path.join(move_dir, task, image))

    print(f'{c} done')
