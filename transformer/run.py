import torch
import torchtext

print("Hi")

dataset = torchtext.datasets.WikiText2(root = '/Users/abhiramkandiyana/LLMsFromScratch/transformer', split=('train', 'valid', 'test'))
print(dataset)


