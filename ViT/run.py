import os

import torch
from model import VisionTransformer
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np

print(matplotlib.get_backend())


def analyze_image_dimensions(root_dir):
    # 1. Create a dataset
    dataset = ImageFolder(root=root_dir, transform=None)

    # 2. Collect dimensions
    widths = []
    heights = []
    for img_path, _ in dataset.imgs:   # dataset.imgs is a list of (path, class_index)
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

    # 3. Plot the distribution of dimensions
    plt.figure()
    plt.scatter(widths, heights)  # x-axis = width, y-axis = height
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Distribution of Image Dimensions')
    plt.show()


def create_patches(img_path, h, w, c, patch_size=16):

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    patches = []

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            patch = patch.reshape(patch_size * patch_size * c)
            patches.append(patch)

    return patches

def input_preprocessing(main_dir_path, w=64, h= 64, c=3, d_patch=16,):
    # img_transform = transforms.Compose([transforms.ToTensor()])
    #
    # dataset = ImageFolder(
    #     root = main_dir_path,
    #     # transform=img_transform
    # )

    class_list = os.listdir(main_dir_path)
    class_list.sort()
    if '.DS_Store' in class_list:
        class_list.remove('.DS_Store')

    dataset = []
    for class_index,class_name in enumerate(class_list):
        img_dir = os.listdir(os.path.join(main_dir_path, class_name))
        for img_index, img_name in enumerate(img_dir):
            img_path = os.path.join(main_dir_path, class_name, img_name)
            patches = create_patches(img_path,h, w, c, d_patch)
            dataset.append((patches,class_index))

    print(dataset)


main_dir_path = '/Users/abhiramkandiyana/LLMsFromScratch/ViT/data'
# analyze_image_dimensions(main_dir_path)
input_preprocessing(main_dir_path)

# d_model = 32
# seq_len = 5
# n_layers = 2
# n_heads = 2
# d_ff = 64
# class_len = 10
#
# model = VisionTransformer(
#     d_model=d_model,
#     seq_len=seq_len,
#     n_layers=n_layers,
#     n_heads=n_heads,
#     d_ff=d_ff,
#     class_len=class_len,
#     pre_training=True
# )
#
# x = torch.tensor([
#     [0, 1, 2, 3, 4],
#     [5, 6, 7, 8, 9]
# ])
#
# logits = model(x)
#
# print("Logits shape:", logits.shape)