import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# tensor = torch.ones((2,4), requires_grad = True, device=device)
#
# res = torch.sum(tensor)
#
# res.backward()
#
# print(tensor.grad)

'''
#NORMALIZE A TENSOR#

tensor = torch.rand((3, 3), device=device)


def normalize(tensor):
    tensor_mean = torch.mean(tensor)
    tensor_std = torch.std(tensor) + 1e-8  # Prevent division by zero
    return (tensor - tensor_mean) / tensor_std



tensor = normalize(tensor)
print(torch.mean(tensor))
print(torch.std(tensor))
'''

'''
BROADCASTING : EITHER THE DIMENSIONS SHOULD BE THE SAME, OR IF ONE OF THE DIMENSIONS IS 1, THEN THE TENSORS ARE COMPATIBLE FOR BROADCASTING

tensor1 = torch.rand(3,1)
tensor2 = torch.rand(1,3)

tensor3 = tensor1 + tensor2

print(tensor3)

'''


'''
GRADIENT CALCULATION : Gradients can only be calculated based upon scalar values

When .backward is called multiple times, the gradients are accumulated
'''

tensor = torch.randn((3,3), requires_grad=True, device=device)

tensor_mean = torch.mean(tensor)

tensor_mean.backward()

print(tensor.grad)





