import os
import random
from PIL import Image

move_dir = './'     # Directory to move images into
data_path = 'AID'   # Path to the dataset

split = [0.9, 0.1]
split_labels = ['train', 'val']

for c in os.listdir(data_path):
    class_path = os.path.join(data_path, c)

    images = os.listdir(class_path)
    selected_images = random.sample(images, len(images))

    start_idx = 0
    for task, fraction in zip(split_labels, split):
        task_dir = os.path.join(move_dir, task) 
        os.makedirs(task_dir, exist_ok=True)
        
        count = int(len(selected_images) * fraction)
        subset = selected_images[start_idx:start_idx + count]
        start_idx += count

        for image in subset:
            src_path = os.path.join(class_path, image)
            dest_path = os.path.join(task_dir, image)
            try:
                os.rename(src_path, dest_path)
            except Exception as e:
                print(f"Error processing {src_path}: {e}")

    print(f"Class '{c}' processed successfully.")
