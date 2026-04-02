import os, shutil, random

train_images = 'dataset/detection/train/images'
train_labels = 'dataset/detection/train/labels'
valid_images = 'dataset/detection/valid/images'
valid_labels = 'dataset/detection/valid/labels'

# Clear existing mismatched valid files
for f in os.listdir(valid_images): os.remove(os.path.join(valid_images, f))
for f in os.listdir(valid_labels): os.remove(os.path.join(valid_labels, f))

# Get train images that have matching labels
paired = []
for img in os.listdir(train_images):
    stem = os.path.splitext(img)[0]
    lbl = stem + '.txt'
    if os.path.exists(os.path.join(train_labels, lbl)):
        paired.append(stem)

# Move 20% to valid
random.seed(42)
val_set = random.sample(paired, int(len(paired) * 0.2))

for stem in val_set:
    # Find the image extension
    for ext in ['.jpg', '.jpeg', '.png']:
        src_img = os.path.join(train_images, stem + ext)
        if os.path.exists(src_img):
            shutil.move(src_img, os.path.join(valid_images, stem + ext))
            break
    shutil.move(os.path.join(train_labels, stem + '.txt'),
                os.path.join(valid_labels, stem + '.txt'))

print(f'Moved {len(val_set)} image+label pairs to valid/')
print(f'Train remaining: {len(paired) - len(val_set)}')