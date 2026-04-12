# fix_labels.py — fixes the train/valid split for the yolo detection dataset
# had mismatched files in valid/ so this script clears and rebuilds it properly
# moves 20% of paired image+label files from train to valid

import os, shutil, random

# paths to the yolo dataset folders
train_images = 'dataset/detection/train/images'
train_labels = 'dataset/detection/train/labels'
valid_images = 'dataset/detection/valid/images'
valid_labels = 'dataset/detection/valid/labels'

# clear out whatever was in valid before — it had mismatched files
for f in os.listdir(valid_images):
    os.remove(os.path.join(valid_images, f))
for f in os.listdir(valid_labels):
    os.remove(os.path.join(valid_labels, f))

# only keep images that have a matching label file
# yolo needs both image and .txt label to be useful
paired = []
for img in os.listdir(train_images):
    stem = os.path.splitext(img)[0]  # filename without extension
    lbl = stem + '.txt'
    if os.path.exists(os.path.join(train_labels, lbl)):
        paired.append(stem)

# randomly pick 20% for validation — seed 42 so its reproducible
random.seed(42)
val_set = random.sample(paired, int(len(paired) * 0.2))

# move selected files from train to valid
for stem in val_set:
    # image could be jpg, jpeg or png — check all three
    for ext in ['.jpg', '.jpeg', '.png']:
        src_img = os.path.join(train_images, stem + ext)
        if os.path.exists(src_img):
            shutil.move(src_img, os.path.join(valid_images, stem + ext))
            break  # found the image, no need to check other extensions

    # move the corresponding label file too
    shutil.move(
        os.path.join(train_labels, stem + '.txt'),
        os.path.join(valid_labels, stem + '.txt')
    )

print(f'Moved {len(val_set)} image+label pairs to valid/')
print(f'Train remaining: {len(paired) - len(val_set)}')