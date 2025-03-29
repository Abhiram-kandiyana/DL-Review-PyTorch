import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from model import VisionTransformer
import math


class PatchDataset(Dataset):
    def __init__(self, data_with_patches):
        """
        data_with_patches is expected to be a list of (patch_tensor, label) pairs
        as returned by build_dataset_with_patches.
        """
        self.data = data_with_patches

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a single (patch_tensor, label) pair.
        patch_tensor: shape [num_patches, patch_dim]
        label: integer class label
        """
        return self.data[idx]

def create_patches_torch(img_tensor, patch_size=16):
    """
    img_tensor: A PyTorch tensor of shape [C, H, W]
    patch_size: Size of each patch (e.g., 16)

    Returns a tensor of shape [num_patches, patch_size * patch_size * C]
    """
    # H and W
    _, H, W = img_tensor.shape

    # (1) Use unfold along the height dimension
    # unfolding => shape [C, (H//patch_size), patch_size, W]
    patches = img_tensor.unfold(dimension=1, size=patch_size, step=patch_size)
    # Now patches shape: [C, (H/patch_size), patch_size, W]

    # (2) Then unfold along the width dimension
    # => shape [C, (H//patch_size), patch_size, (W//patch_size), patch_size]
    patches = patches.unfold(dimension=3, size=patch_size, step=patch_size)
    # Now patches shape: [C, (H/patch_size), patch_size, (W/patch_size), patch_size]

    # (3) Re-arrange to group the patch dimensions in front
    # => [ (H/patch_size) * (W/patch_size), C, patch_size, patch_size ]
    patches = patches.permute(1, 3, 0, 2, 4)
    # shape is now [(H/patch_size), (W/patch_size), C, patch_size, patch_size]
    patches = patches.reshape(-1, img_tensor.size(0), patch_size, patch_size)

    # (4) Flatten each patch to [patch_size * patch_size * C]
    patches = patches.reshape(patches.size(0), -1)

    return patches


def build_dataset_with_patches(main_dir_path, patch_size=16):
    """
    1. Loads images from main_dir_path via ImageFolder.
    2. Resizes them to 64x64, converts to tensor.
    3. Splits each image into patch_size x patch_size patches.
    4. Flattens patches into 1D vectors.
    5. Returns a list of (patch_tensor, label) pairs for each image.

    NOTE: This example stores everything in memory. If your dataset is huge,
          consider building a custom Dataset or a streaming approach to avoid
          storing all patches at once.
    """

    # Basic transform: resize to 64x64, convert to tensor
    transform_ops = transforms.Compose([
        # transforms.Resize((64, 64)),
        transforms.ToTensor(),  # shape => [C, 64, 64]
    ])

    # 1. Build dataset using ImageFolder
    dataset = datasets.ImageFolder(root=main_dir_path, transform=transform_ops)

    # We'll create a list to hold (patch_tensor, label) for each image
    data_with_patches = []

    # 2. Iterate through dataset, create patches
    for idx in range(len(dataset)):
        img_tensor, label = dataset[idx]
        # shape => [C=3, H=64, W=64] if it's a color image

        # 3. Create patches for this image
        patch_tensor = create_patches_torch(img_tensor, patch_size=patch_size)
        # shape => [num_patches, patch_size * patch_size * C]

        # 4. Store the patch_tensor + label
        data_with_patches.append((patch_tensor, label))

    return data_with_patches

def createBatches(data_with_patches, batch_size=16, shuffle=True, num_workers=0):
    """
    Takes a list of (patch_tensor, label) pairs and returns
    a PyTorch DataLoader that yields mini-batches.
    """
    dataset = PatchDataset(data_with_patches)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return data_loader




if __name__ == '__main__':
    # Example usage:
    main_dir = '/Users/abhiramkandiyana/LLMsFromScratch/ViT/data'  # each subfolder in 'images/' is a class
    d_model = 16
    n_layers = 8
    n_heads = 8
    d_ff = 512
    class_len = 2
    pre_training = True
    patch_size = 16
    batch_size = 16

    # Build dataset with patches
    data_with_patches = build_dataset_with_patches(main_dir, patch_size)

    dataset = PatchDataset(data_with_patches)

    seq_len = dataset.data[0].shape[1]

    dataloader = createBatches(data_with_patches, batch_size=batch_size)

    model = VisionTransformer(d_model, seq_len, n_layers, n_heads, d_ff, class_len, pre_training)

    for patch_batch, label_batch in dataloader:

        pass



    # # Print example shapes
    # print(f'Total items in dataset: {len(data_with_patches)}')
    # example_patch_tensor, example_label = data_with_patches[0]
    # print(f'Example patch tensor shape: {example_patch_tensor.shape}')
    # print(f'Example label: {example_label}')