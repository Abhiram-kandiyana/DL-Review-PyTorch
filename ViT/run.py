import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from model import VisionTransformer
from torch import nn
import torch.optim as optim
import matplotlib
from plots import loss_curve
import matplotlib.pyplot as plt

seed = 42


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

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8),
                                                                          int(len(dataset) * 0.2)])
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * 0.8),
                                                                               int(len(train_dataset) * 0.2)])

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return train_data_loader, val_data_loader, test_data_loader


def train(train_loader, val_loader, model, epochs=100, lr=0.005, device=torch.device('cpu')):
    model = model.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_loss_arr = []
    val_loss_arr = []

    for epoch in range(epochs):

        model.train()
        train_loss_epoch = 0.0

        for patch_batch, label_batch in train_loader:
            patch_batch = patch_batch.to(device)
            label_batch = label_batch.to(device)

            logits_batch = model(patch_batch)

            batch_loss = loss(logits_batch, label_batch)
            batch_loss.backward()

            optimizer.step()

            train_loss_epoch += batch_loss.item()

        model.eval()
        val_loss_epoch = 0.0

        with torch.no_grad():
            for patch_batch, label_batch in val_loader:
                patch_batch = patch_batch.to(device)
                label_batch = label_batch.to(device)

                logits_batch = model(patch_batch)

                batch_loss = loss(logits_batch, label_batch)

                val_loss_epoch += batch_loss.item()

        train_loss_epoch = train_loss_epoch / len(train_loader)
        val_loss_epoch = val_loss_epoch / len(val_loader)

        print(
            f'Epoch {epoch + 1} completed. loss: {train_loss_epoch}, validation loss : {val_loss_epoch}')

        train_loss_arr.append(train_loss_epoch)
        val_loss_arr.append(val_loss_epoch)

    #TODO : Add code to save the plot at a user defined path
    loss_curve(train_loss_arr, val_loss_arr)








if __name__ == '__main__':
    # Example usage:
    main_dir = '/Users/abhiramkandiyana/LLMsFromScratch/ViT/data'  # each subfolder in 'images/' is a class
    # Model Params
    d_model = 384
    n_layers = 6
    n_heads = 4
    d_ff = 1536
    class_len = 2
    pre_training = True
    n_patches = 16

    #Hyper Params
    batch_size = 16
    lr = 0.005

    # Build dataset with patches
    data_with_patches = build_dataset_with_patches(main_dir, n_patches)

    seq_len = data_with_patches[0][0].shape[0]
    patch_dim = data_with_patches[0][0].shape[1]

    train_loader, val_loader, test_loader = createBatches(data_with_patches, batch_size=batch_size)

    model = VisionTransformer(d_model, seq_len, n_layers, n_heads, d_ff, class_len, patch_dim, pre_training)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(train_loader, val_loader, model, lr, device=device)

    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for patch_batch, label_batch in test_loader:
            logits = model(patch_batch)
            preds = logits.argmax(dim=1)
            correct = (preds == label_batch).sum().item()
            total_correct += correct
            total_samples += label_batch.size(0)

    accuracy = total_correct / total_samples
    print(f'Test Accuracy: {accuracy:.2f}')

    # # Print example shapes
    # print(f'Total items in dataset: {len(data_with_patches)}')
    # example_patch_tensor, example_label = data_with_patches[0]
    # print(f'Example patch tensor shape: {example_patch_tensor.shape}')
    # print(f'Example label: {example_label}')
