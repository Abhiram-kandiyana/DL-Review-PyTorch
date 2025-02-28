import torch

tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([7, 8, 9])

stacked_tensor_dim0 = torch.stack((tensor1, tensor2), dim=1)
print("Stacked along dimension 0:")
print(stacked_tensor_dim0)
print("Shape:", stacked_tensor_dim0.shape)
# Expected output:
# Stacked along dimension 0:
# tensor([[[ 1,  2,  3],
#          [ 4,  5,  6]],
#
#         [[ 7,  8,  9],
#          [10, 11, 12]]])
# Shape: torch.Size([2, 2, 3])


stacked_tensor_dim1 = torch.stack((tensor1, tensor2), dim=1)
print("\nStacked along dimension 1:")
print(stacked_tensor_dim1)
print("Shape:", stacked_tensor_dim1.shape)
# Expected output:
# Stacked along dimension 1:
# tensor([[[ 1,  2,  3],
#          [ 7,  8,  9]],
#
#         [[ 4,  5,  6],
#          [10, 11, 12]]])
# Shape: torch.Size([2, 2, 3])