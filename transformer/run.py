import torch
import torchtext
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_data = dataset["train"]["text"]
valid_data = dataset["validation"]["text"]
test_data = dataset["test"]["text"]

print(train_data[:5])

tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens = ["UNK", "PAD", "MASK"])

tokenizer.train(train_data, trainer=trainer)


