# %% [markdown]
# # Deep Learning for NLP - Exercise 01
# Building, Tuning and Evaluating a standard RNN Model
# 
# General hints:
# * Have a look at the imports below when solving the tasks
# * Use the given modules and all submodules of the imports, but don't import anything else!
#     * For instance, you can use other functions under the `torch` or `nn` namespace, but don't import e.g. PyTorch Lightning, etc.
# * It is recommended to install all packages from the provided environment file
# * Feel free to test your code between sub-tasks of the exercise sheet, so that you can spot mistakes early (wrong shapes, impossible numbers, NaNs, ...)
# * Just keep in mind that your final submission should be compliant to the provided initial format of this file
# 
# Submission guidelines:
# * Make sure that the code runs on package versions from the the provided environment file
# * Do not add or change any imports (also don't change the naming of imports, e.g. `torch.nn.functional as f`)
# * Remove your personal, additional code testings and experiments throughout the notebook
# * Do not change the class, function or naming structure as we will run tests on the given names
# * Additionally export this notebook as a `.py` file, and submit **both** the executed `.ipynb` notebook with plots in it **and** the `.py` file
# * **Deviation from the above guidelines will result in partial or full loss of points**

# %% [markdown]
# If you are using Google Colab or similar services, make sure to install all necessary packages so that the import cell below is working.
# 
# Usually, you would need to `!pip install`:
# ```
# !pip install datasets==3.0.1
# !pip install spacy==3.6.1
# !pip install torch==2.0.1    # just to be sure we are all working with the same version
# !pip install torchtext==0.15.2
# !python -m spacy download en_core_web_sm
# ```
# 
# Make sure to comment out the lines before submitting!

# %%
import os
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Sampler, BatchSampler, Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

from datasets import load_dataset

# %%
def regularized_f1(train_f1, dev_f1, threshold=0.0015):
    """
    Returns development F1 if overfitting is below threshold, otherwise 0.
    """
    return dev_f1 if (train_f1 - dev_f1) < threshold else 0


def save_metrics(*args, path, fname):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isfile(path + fname):
        with open(path + fname, "w", newline="\n") as f:
            f.write(
                ",".join(
                    [
                        "config",
                        "epoch",
                        "train_loss",
                        "train_acc",
                        "train_f1",
                        "val_loss",
                        "val_acc",
                        "val_f1",
                    ]
                )
            )
            f.write("\n")
    if args:
        with open(path + fname, "a", newline="\n") as f:
            f.write(",".join([str(arg) for arg in args]))
            f.write("\n")

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(1234)

# %%
VOCAB_SIZE = 20_000
BATCH_SIZE = 32
NUM_EPOCHS = 15
MAX_LEN = 256
LEARNING_RATE = 1e-4

# %% [markdown]
# ## Task 1 - Download and prepare the dataset
# - Load the train and test set of IMDB (it is included in the `datasets` module we imported above)
# - Split the train set into train and validation set
#     * Train set should consist of the middle 10% to 85% of data
#     * Validation set should be the two remaining ends
#     * You can achieve this slicing directly within the `load_dataset` function, check out the [Huggingface slicing API](https://huggingface.co/docs/datasets/v2.13.1/loading#slice-splits)
# - Test set should stay unchanged

# %%
# load dataset in splits
train_data = load_dataset("imdb", split="train[10%:85%]")
dev_data = load_dataset("imdb", split="train[:10%]+train[85%:]")
test_data = load_dataset("imdb", split="test")

# %% [markdown]
# * Define the tokenizer using `get_tokenizer` with spacy's `en_core_web_sm` module
#     * You don't have to import spacy for that, but it is necessary to have spacy installed and the `en_core_web_sm` module downloaded
# * Create the vocabulary using `build_vocab_from_iterator`
#     * Think about which split(s) should be used to build the vocabulary
#     * Include two special tokens: `'<UNK>'` at index `0`, `'<PAD>'` at index `1`
#     * Limit the vocab size to `VOCAB_SIZE`, as defined in the beginning
#     * Set the vocab's default returning index to `0` by making the `'<UNK>'` token default
#     
# Hint:
# * This might be a good moment to add a personal test to check whether your vocab actually returns `0` for an unknown input token

# %%
# define tokenizer
tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

# define vocabulary
vocab = build_vocab_from_iterator(
    (tokenizer(text["text"]) for text in train_data),
    max_tokens=VOCAB_SIZE,
    specials=['<UNK>', '<PAD>']
)

vocab.set_default_index(vocab['<UNK>'])

# %% [markdown]
# * Use the tokenizer and vocabulary to turn your three data splits into indices
# * Limit the maximum tokenized sequence length to `MAX_LEN`
# 
# **Note**:
# In practice, performing this step on its own during the preprocessing stage is usually not feasible due to the memory constraints associated with storing large datasets. Consequently, the tokenization and indexing steps are typically performed "on the fly" within the `DataLoader`, specifically in the `collate_batch` step.

# %%
train_idx = [vocab(tokenizer(text["text"])) if len(tokenizer(text["text"])) <= MAX_LEN else vocab(tokenizer(text["text"])[0:MAX_LEN]) for text in train_data]
dev_idx = [vocab(tokenizer(text["text"])) if len(tokenizer(text["text"])) <= MAX_LEN else vocab(tokenizer(text["text"])[0:MAX_LEN]) for text in dev_data]
test_idx = [vocab(tokenizer(text["text"])) if len(tokenizer(text["text"])) <= MAX_LEN else vocab(tokenizer(text["text"])[0:MAX_LEN]) for text in test_data]

# %% [markdown]
# * Define a torch dataset by inhereting from `Dataset`
# * It should create the building block to return the tokenized indices and labels for a given index
# * Instantiate it

# %%
class ImdbDataset(Dataset):
    def __init__(self, seq, lbl):
        self.seq = seq
        self.lbl = lbl

    def __getitem__(self, idx):
        return self.seq[idx], self.lbl[idx]

    def __len__(self):
        return len(self.seq)

# %%
train_set = ImdbDataset(train_idx, train_data['label'])

# %% [markdown]
# * Having batches in which samples have a similar length, and thus less padding variations, improves training results
# * A `GroupedSampler` allows us to create a sampler with which we can customize the data loading process
# * It can then be implemented into the `DataLoader`, which automates loading data in multiple processes
# * Write a sampler which allows us to group together samples of similar length into a batch
#     * The `GroupedSampler` takes as input the tokenized sequences from `ImdbDataset`, as well as the batch size
#     * First, in the `__init__` method, pair each sequence index with its tokenized sequence length
#         * The result should be a list of tuples: `[(index, tokenized_sequence_length), ...]`
#     * In the `__iter__` method, we now:
#         * Shuffle the list
#         * generate groups of size `BATCH_SIZE * 100`
#         * Each group of size `BATCH_SIZE * 100` should be sorted in itself by the sequence length we calculated above
#             * Sorting within each group is important because sorting based on the whole dataset would eliminate all training input variations
#             * By shuffling in the `__iter__` method, we shuffle the set of indices in each new iteration (which equals an epoch), therefore, we keep input variation
#         * The result should be a list of tuples sorted by ascending sequence length: `[(index, tokenized_sequence_length), ...]`
#         * After each `BATCH_SIZE * 100` number of tuples, the sequence length of samples should drop and increase again
#         * Example:
#             ```
#             Sample index 3199: (1234, 256)
#             Sample index 3200: (567, 32)
#             Sample index 3201: (890, 33)
#             ```
#         * Filter the created and sorted list to only consist of indices. Make sure to keep the sorting!
#         * Return this list as an iterator
#     * Complete the `__len__` method

# %%
class GroupedSampler(Sampler):
    def __init__(self, seqs, batch_size):
        self.seqs = seqs
        self.batch_size = batch_size
        # pairing each sequence index with its length
        self.index_length = [(i, len(seq)) for i, seq in enumerate(self.seqs)]

    def __iter__(self):
        # shuffle the indices in each iteration
        random.shuffle(self.index_length)
        group_size = self.batch_size * 100

        grouped_indices = list()
        # take batch_size * 100 indices at a time as a slice
        for i in range(0, len(self.index_length), group_size):
            group = self.index_length[i:i+group_size]
            group = sorted(group, key=lambda x: x[1]) # sort by length within group
            grouped_indices.extend([idx for idx, _ in group]) # append indices to grouped_indices

        return iter(grouped_indices) # create an iterator from grouped_indices

    def __len__(self):
        return len(self.seqs)

# %% [markdown]
# * Now create the `GroupedSampler`, use it as input to create a `BatchSampler` (imported in the beginning)

# %%
train_grouped_sampler = GroupedSampler(train_idx, BATCH_SIZE)
train_sampler = BatchSampler(train_grouped_sampler, BATCH_SIZE, False)

# %% [markdown]
# * Define a collate function which takes in a `batch` of tokenized sequences and labels created by the `BatchSampler`
#     * Make sure to understand the structure of an input `batch`. Test around a bit to see what exactly they return.
# * The collate function then:
#     * pads these indices to the same length
#         * use `padding_value=1`, `0` should be reserved for `UNK` token
#     * turns the labels into tensors
#     * finally, it creates a tensor which stores the length of all tokenized sentences **before** padding
#     * the function should return 3 batched tensors: sequences, labels, lengths

# %%
# define collate function
def collate_batch(batch):
    seqs, labels = zip(*batch)
    lens_before_padding = torch.tensor([len(seq) for seq in seqs])
    seqs = pad_sequence([torch.tensor(seq) for seq in seqs], padding_value=1, batch_first=True)
    return seqs, torch.tensor(labels), lens_before_padding

# %% [markdown]
# * Now create the final `DataLoader` for the train set
#     * For your training, set the number of workers to your liking/cpu cores setup
#     * When submitting this exercise, please set `num_workers=2` at maximum
# * Repeat the `DataLoader` creation process for the validation and test set
#     * It is not necessary to introduce randomness into the validation and test set
#     * Create an `ImdbDataset` and `DataLoader` instance
#     * leave `shuffle` off and don't include any Samplers
#     * still include the correct batch size and collate function

# %%
# create dataloaders
train_loader = DataLoader(
    num_workers=0, # to be changed upon submission
    dataset=train_set,
    batch_sampler=train_sampler,
    collate_fn=collate_batch
)

validation_loader = DataLoader(
    num_workers=0,
    dataset=ImdbDataset(dev_idx, dev_data['label']),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch
)

test_loader = DataLoader(
    num_workers=0,
    dataset=ImdbDataset(test_idx, test_data['label']),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch
)

# %% [markdown]
# ## Task 2 - Build Your Model
# The model should consist of:
# - an **embedding layer**, which takes `vocab_size` and `embedding_dim` as parameters
# - a **dropout layer**, which takes `dropout` as a parameter
# - an **LSTM layer**, which takes `embedding_dim` and `rnn_size` as parameters, and is bidirectional
# - a **linear layer**, which takes the dimension of rnn output as input dimension and returns an output of `hidden_size` dimensions
# - a **linear layer**, which first takes the previous layers output as input and returns one prediction per class of the dataset
# - the output of the BiLSTM has hidden representation tensors for each index of each sequence. However, for the task of sequence classification, we just need one hidden representation tensor per sequence. Use `torch.mean()` as a pooling function for dimensionality reduction.
# - use **dropout** on the embeddings and appropriate linear layer
# - use **ReLU** as the activation function on the appropriate linear layer
# 
# _Hints:_
#   - keep the position of the batch dimension equal across all layers
#   - _use `pack_padded_sequence`_ and `pad_packed_sequence` at the appropriate steps. For more information, check out [this answer on stackoverflow](https://stackoverflow.com/a/56211056)
#   - remember to include the `padding_idx=1` at relevant positions
#   - as this is a binary classification task, it is possible to have 1 or 2 output neurons. Use your preference, but adjust the loss function towards your choice

# %%
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_size, hidden_size, dropout):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=1,
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.LSTM = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=rnn_size,
            bidirectional=True,
            batch_first=True
        )
        self.linear = nn.Linear(in_features=rnn_size * 2, out_features=hidden_size)
        self.linear_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, seq, lengths):
        embedded = self.embedding(seq)
        embedded = self.embedding_dropout(embedded)

        # now we can pack the padded sequence
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        # passing through the LSTM
        packed_output, (_, _) = self.LSTM(packed)

        # unpack the packed sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # apply mean pooling
        mean_pooled = torch.mean(output, dim=1)

        # passing through the linear layer
        linear_output = self.linear(mean_pooled)
        linear_output = self.relu(linear_output)
        linear_output = self.linear_dropout(linear_output)

        # final linear layer
        logits = self.fc(linear_output)

        return logits

# %% [markdown]
# ## Task 3 - Inner train loop
# * Create a global `device` variable which checks whether a GPU is available or not, and sets the device to either GPU or CPU.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# * Write the inner train/test loop by completing the function `process`.
#     * It takes the model, a dataloader, criterion and optionally the optimizer
#     * The function iterates once through the dataloader, i.e. one epoch
#     * Include the `tqdm` functionality for the loop through the dataloader by placing the loader inside `tqdm()`
#         * Print its output to `file=sys.stdout`, and use `'batches'` as unit
#         * You can also add a `desc='...'` to get a marking whether we currently train or evaluate
#     * The function also moves the sequences and labels to `device`
#     * The `lengths` need to stay on CPU!
#     * If the optimizer is given, training with backpropagation is performed, then the below defined metrics are returned
#     * If the optimizer is missing, evaluation is performed and the below described metrics are calculated
#     * Values to be calculated:
#         * Loss, Accuracy, both as averages of the total number of samples per epoch
#         * F1 score between all predictions and labels of the epoch

# %%
def process(model, loader, criterion, optim=None):
    # Initialize tqdm for progress tracking
    loop = tqdm(loader, file=sys.stdout, unit='batches', desc="Training" if optim else "Evaluating")

    # move model to device
    model.to(device)
    if optim:
        model.train()
    else:
        model.eval()

    # init
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch in loop:
        seqs, labels, lens = batch
        seqs, labels, lens = seqs.to(device), labels.to(device), lens.to('cpu')

        if optim:
            y_hat = model(seqs, lens)
            ### DEBUGGING
            ###print(f'Number of sequences {len(seqs)}')
            ###print(f'Shape of y_hat {y_hat.squeeze().shape}')
            ###print(f'Shape of labels {labels.float().shape}')
            batch_loss = criterion(y_hat.squeeze(), labels.float())

            optim.zero_grad()
            batch_loss.backward()
            optim.step()

        else:
            with torch.no_grad():
                y_hat = model(seqs, lens)
                batch_loss = criterion(y_hat.squeeze(), labels.float())

        total_loss += batch_loss.item() * labels.size(0)
        preds = (y_hat.squeeze() > 0.5).int()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / total
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds)

    return avg_loss, accuracy, f1

# %% [markdown]
# # Task 4 - Training and Hyperparameter Optimization
# In the following, we provide 3 configurations for the above created BiLSTM. Try to understand how they differ from each other.

# %%
configs = {
    "config1": {
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": 10,
        "hidden_size": 10,
        "rnn_size": 10,
        "dropout": 0.5
    },
    "config2": {
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": 64,
        "hidden_size": 32,
        "rnn_size": 256,
        "dropout": 0.5
    },
    "config3": {
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": 300,
        "hidden_size": 256,
        "rnn_size": 256,
        "dropout": 0.5
    }

}

# %% [markdown]
# * Choose the correct criterion to train and evaluate your created model

# %%
criterion = nn.BCEWithLogitsLoss()

# %% [markdown]
# Use the given functions `regularized_f1` and `save_metrics` from the start of the notebook to implement the hyperparameter search and training runs.
# 
# Specifically:
# * Iterate through each configuration
# * Create and re-create the model for each new configuration run
#     * Move the model to the `device`
# * Create and re-create the optimizer with each new configuration model's paramaters
#     * Use Adam as the optimizer
#     * Use the learning rate defined at the beginning of notebook
# * Train each configuration for `NUM_EPOCHS` epochs
# * Change the model into train and evaluation mode at appropriate times
# * Stop gradient calculation for evaluation runs
# * Save metrics after each train and evaluation runs.
#     * Have a look at the function to see what the expected inputs are
#     * In the `.csv` file, only numbers should be entered
#     * For instance, the inputs for the columns `config` and `epoch` should be e.g. `1`, _not_ `config1` or `epoch1`
# * Optional: Print training progress for your own information
# 
# In order to check whether our model generalizes or just 'remembers', we need to compare the model's performance on the train set to the performance on the validation set. As we are only interested in non-overfitting performances, we only want to save model checkpoints when the model actually generalizes, i.e. has a higher F1 score on the validation set than on the train set.
# * Calculate the regularized f1 score using the given function
# * Keep track of multiple values during training:
#     * Save the overall (i.e. across all configs *and* epochs) highest validation F1 score
#         * Save your best model parameters
#         * Overwrite your model parameters every time your model fulfills both the `regularized_f1` criteria and is better than the previous overall highest F1 score
#         * In the end, the last saved `best_model.pt` parameters are automatically the best
#         * Hint: Keep track (e.g. by printing or in a variable), which config produced the best model, so you can directly load that config for the test set run.
#     * Track the highest F1 score inside a configuration but across epochs
#         * Implement early-stopping for a configuration run if 3 consecutive epochs are below the highest F1 score for the current configuration

# %%
path = './'
logging_file = 'results.csv'
best_overall_f1 = 0
best_params = None
best_config = None
seed_everything(1234)

for config, paramset in configs.items():
    best_f1_within_conf = 0
    epochs_below = 0

    model = BiLSTM(
        vocab_size=paramset['vocab_size'], 
        embedding_dim=paramset['embedding_dim'],
        rnn_size=paramset['rnn_size'],
        hidden_size=paramset['hidden_size'],
        dropout=paramset['dropout']
        )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        print(f'__Epoch {epoch} for {config}__')
        train_loss, train_acc, train_f1 = process(model, train_loader, criterion=criterion, optim=optimizer)
        val_loss, val_acc, val_f1 = process(model, validation_loader, criterion=criterion, optim=None)

        # Save metrics to file
        save_metrics(
            config[-1],
            epoch,
            train_loss,
            train_acc,
            train_f1,
            val_loss,
            val_acc,
            val_f1,
            path=path,
            fname=logging_file
        )

        # Print the results of epoch
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Early stopping
        if val_f1 > best_f1_within_conf:
            print(f'New best F1 within config: {val_f1}')
            best_f1_within_conf = val_f1
            epochs_below = 0
        elif val_f1 <= best_f1_within_conf and best_f1_within_conf != 0:
            epochs_below += 1
            print(f'F1 not improved within config for {epochs_below} epochs')
        
        if epochs_below >= 3:
            print(f'Early stopping at epoch {epoch} for {config}')
            break
                
        reg_f1 = regularized_f1(train_f1, val_f1)
        # Save global best
        if reg_f1 > 0 and reg_f1 > best_overall_f1:
            print(f'New best global F1: {reg_f1} found and saved!')
            best_overall_f1 = val_f1
            best_params = model.state_dict()
            best_config = config
            best_epoch = epoch
            torch.save(best_params, "best_model.pt")

        

# %% [markdown]
# * Load in the created `results.csv` file
# * Create 6 plots: 3 rows with each 2 sub-plots
# * Each row of plots should correspond to a configuration
# * Each left plot shows the loss progression per epoch
#     * Include both train and evaluation progress in the same plot, but plot the evaluation lines dashed
#     * Plot losses in blue
# * Each right plot shows both the accuracy and F1 progression per epoch
#     * Include both train and evaluation progess in the same plot, but plot the evaluation lines dashed
#     * Plot accuracy in orange, and plot F1 in green
# * Have a look at the example plot file included with this exercise. It constitutes one row of the plot.
# * After plotting, briefly describe what problems and successes you see with each configuration

# %% [markdown]
# **Here you can see a partial example plot**
# ![example_plot.png](data:image/png;base64,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CX799dcYtb99+zYNGzakatWqnD17luHDh/Ptt9/y999/J3KkaYeREZQoAd9+C+vXw9OncP48zJr1fkqLt29h1y4YMQKqVNESxLp1YcIEOHQIgoIM/SiEEMIwBgwYwMKFC1m8eDGXL1+mf//++Pj40KOHNvxr2LBhdOrUKdJ6ixYtokKFChQtGnkC5++++46dO3fy008/ceXKFX766Sd2795Nv379EvvhiBi6efMmVapUwcnJCXt7e0ArMvTll1/Ss2dPA0dnQL5X4fp/P9yXnibDRaOh0+nYWsWdv7LANwF74XA7CIvll6nQt3DwC7ixANBB2TlQYkKym9ojT4Y8rGu9jjv97pA1XdaYreR/Bza7wLU5KTIx/Hrz19RZXgfvV96GDkWjkjFA/fPPP59sM3jwYFWwYMEIy/73v/+pihUrxng/r1+/VoB6/fp1XMJM88LClLpwQanZs5X64gulMmVSSnt3vr9YWipVu7ZS48YpdeCAUoGBho5aCJEaJdfP8zlz5ignJydlZmamSpcurTw8PPT3de7cWVWvXj1C+1evXilLS0u1YMGCaLf5119/qQIFCihTU1NVsGBB9ffff8cqpk89VwEBAcrLy0sFBATEapvJQXh4uPrpp59Unjx5lIWFhSpevLj666+/lFJKhYWFqezZs6t58+ZFWOf06dMKUDdv3lRKKTVt2jRVtGhRZWVlpXLkyKF69uyp3rx5o2+/ZMkSZWdnF20MQITL6NGjI9z/ufXjI9m/dvubKrUSpfY1NnQkKcPtVUqtNtWes101lAp6FbP1Ap8ptcNVW2+1uVI+6xM3zqR2orf22HbXNnQksbbea71iDMporJE6fu94gmwzvse+FD/zXHSltxctWkRISAimppG7xIOCggj6oMtKKq3Fj5ERFC2qXfr00VJALy/Yvx88PLS/T5/Cnj3aBcDMDPLmhfz5I1+cnGRORCFE6tKrVy969eoV5X1Lly6NtMzOzo63b99+cputWrWiVaukL0TgH+wf7X3GRsZYmFjEqK2RzghLU8vPto3tUKoffviB9evXM2/ePJydnTlw4ABfffUVDg4OVK9enbZt27Jy5Up9Dy3AqlWrcHV1JW9erYKjkZERs2bNInfu3Ny+fZtevXoxePBg5s6dG6MYHj58SJ06dahfvz4DBw7Exsbw52olC4/3w/1NoDOGUj8bOppk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ZtfnvhQfx1RSHXDX26RJk6yO5hA09Pb9LVq0yChVqpTh7u5uFC9e3Bg3bpzVkexSRESE0adPH6NAgQKGh4eHUahQIWPIkCFGVFSU1dEst3r16rt+RnXo0MEwDHP47ffff9/IlSuX4e7ubtSoUcPYs2ePtaHF7unc93h07rs/nfuSRue+e7Pnc5/NMAwj5ctOERERERERsTe6hlBERERERCSdUkEoIiIiIiKSTqkgFBERERERSadUEIqIiIiIiKRTKghFRERERETSKRWEIiIiIiIi6ZQKQhERERERkXRKBaGIiIiIiEg6pYJQRO7LZrMxf/58q2OIiIikGp37JD1RQShix1555RVsNtsdt7p161odTUREJEXo3CeSulysDiAi91e3bl0mTZqUaJ27u7tFaURERFKezn0iqUcthCJ2zt3dnVy5ciW6ZcmSBTC7tIwdO5Z69erh6emJn58fs2fPTrT/nj17qFmzJp6enmTLlo0uXbpw9erVRNtMnDiRkiVL4u7uTu7cuenZs2eixy9cuMALL7xAhgwZKFKkCAsXLkzZFy0iIumazn0iqUcFoYiDe/fdd2nWrBm7du2ibdu2tG7dmv379wNw/fp16tatS5YsWdiyZQuzZ89m5cqViU56Y8eOpUePHnTp0oU9e/awcOFCChcunOg5PvjgA1q0aMHu3bupX78+L7/8MpcuXUrV1ykiInKLzn0iycgQEbvVoUMHw9nZ2ciYMWOi27BhwwzDMAzA6Nq1a6J9KlWqZHTr1s0wDMMYN26ckSVLFuPq1asJj//666+Gk5OTERYWZhiGYeTJk8cYMmTIPTMAxjvvvJNw/+rVq4bNZjOWLl2abK9TRETkFp37RFKXriEUsXPPPfccY8eOTbQua9asCcuBgYGJHgsMDGTnzp0A7N+/nzJlypAxY8aEx6tWrUp8fDwHDx7EZrNx5swZatWqdd8MpUuXTljOmDEjmTJl4ty5c4/6kkRERO5L5z6R1KOCUMTOZcyY8Y5uLA9is9kAMAwjYflu23h6eibpeK6urnfsGx8f/1CZREREkkrnPpHUo2sIRRzcH3/8ccf94sWLA+Dv78/OnTu5du1awuMbNmzAycmJokWLkilTJgoWLMhvv/2WqplFREQeh859IslHLYQidi4qKoqwsLBE61xcXMiePTsAs2fPpnz58lSrVo2pU6eyefNmJkyYAMDLL7/M+++/T4cOHRg6dCjnz5+nV69etGvXDh8fHwCGDh1K165dyZkzJ/Xq1SMyMpINGzbQq1ev1H2hIiIi/9C5TyT1qCAUsXPLli0jd+7cidYVK1aMAwcOAOYoaDNmzKB79+7kypWLqVOn4u/vD0CGDBlYvnw5ffr0oUKFCmTIkIFmzZoxcuTIhGN16NCBmzdv8r///Y8BAwaQPXt2mjdvnnovUERE5D907hNJPTbDMAyrQ4jIo7HZbMybN4+mTZtaHUVERCRV6Nwnkrx0DaGIiIiIiEg6pYJQREREREQknVKXURERERERkXRKLYQiIiIiIiLplApCERERERGRdEoFoYiIiIiISDqlglBERERERCSdUkEoIiIiIiKSTqkgFBERERERSadUEIqIiIiIiKRTKghFRERERETSKRWEIiIiIiIi6ZQKQhERERERkXRKBaGIiIiIiEg6pYJQREREREQknVJBKCIiIiIikk6pIBQREREREUmn/g+lJmf+qUdmAQAAAABJRU5ErkJggg==)
# 

# %%
df = pd.read_csv('results.csv')

# %%
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(3, 2, figsize=(9, 11), sharex=True)

for i in range(1,4):
    df_graph = df.query(f'config == {i}')
    # loss
    ax[i-1,0].plot(df_graph['epoch'], df_graph['train_loss'], color='blue')
    ax[i-1,0].plot(df_graph['epoch'], df_graph['val_loss'], color='blue', linestyle='--')
    ax[i-1,0].set_title(f'Config {i} Loss Progression')
    ax[i-1,0].set_xlabel('Epoch')
    ax[i-1,0].set_ylabel('Loss')
    ax[i-1,0].legend(['Train Loss', 'Val Loss'], loc='upper right')
    # progress
    ax[i-1,1].plot(df_graph['epoch'], df_graph['train_acc'], color='orange')
    ax[i-1,1].plot(df_graph['epoch'], df_graph['val_acc'], color='orange', linestyle='--')
    ax[i-1,1].plot(df_graph['epoch'], df_graph['train_f1'], color='green')
    ax[i-1,1].plot(df_graph['epoch'], df_graph['val_f1'], color='green', linestyle='--')
    ax[i-1,1].set_title(f'Config {i} Accuracy and F1 Progression')
    ax[i-1,1].set_xlabel('Epoch')
    ax[i-1,1].set_ylabel('Metric')
    ax[i-1,1].legend(['Train Acc', 'Val Acc', 'Train F1', 'Val F1'], loc='upper left')
    ax[i-1,1].set_ylim(0, 1)


plt.tight_layout()
plt.show()


# %% [markdown]
# ___
# 
# Student answer here:
# 
# ## Config 1
# - The improvement in learning is quite slow due to the less complexity of the model. The model is underfitting.
# - In parallel with that, loss function initially stagnates and there is no significant progress in validation loss. 
# - Starting from 10th epoch, both F1 and accuracy increases. Yet they're still far lower than the other configs.
# 
# ## Config 2
# - Validation Loss fluctuates but shows a gradual downward trend. Fluctuation might be related to our batch decisions. 
# - Compared to Config 1, Accuracy significantly increases (from ~0.45 to ~0.8).
# - Similarly to Accuracy, we can say that F1 improves more quickly. Having higher validation F1 indicates better generalization performance.
# - Yet, starting from 4th epoch, the gap between training and validation losses increases. Which is a sign of overfitting to training data.
# 
# ## Config 3
# - Apparently, the model overfits from the beginning as having higher validation loss.
# - Both accuracy and F1 are lower for training data. 
# - The curves indicate instable learning process, related to overcomplexity of the model.
# ___

# %% [markdown]
# * As the final step, instantiate a model with the config of the best run
# * Load the `state_dict`
# * Evaluate it on the test set
#     * Don't forget to actviate evaluation mode and deactivate gradient calculation
# * Comment on the performance
#     * Did it generalize well? Why? Why not?
#     * What could be done to improve the performance even further?
#         * Consider also the hyperparameters from the third cell and discuss potential tradeoffs.

# %%
# evaluate on test set
paramset = configs['config2']

model = BiLSTM(
    vocab_size=paramset['vocab_size'], 
    embedding_dim=paramset['embedding_dim'],
    rnn_size=paramset['rnn_size'],
    hidden_size=paramset['hidden_size'],
    dropout=paramset['dropout']
    )


model.load_state_dict(torch.load("best_model.pt"))
model.to(device)

test_loss, test_acc, test_f1 = process(model, test_loader, criterion=criterion, optim=None)

print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")

# %% [markdown]
# ___
# 
# Student answer here:
# - Test loss and performance metrics are relatively similar to train and validation metrics. This is a good sign in terms of generalization. The model classifies well with the unseen data.
# - To improve our model, we can try different configs, searching values between config 2 and config 3. Playing around different embedding dim and hidden size might be a good start.
# - To the models which are more complex than config 2, we can introduce new regularization methods (i.e. weight decay, gradient clipping, etc.)
# - Regarding to the hyperparameters from the third cell:
#     - Larger vocab size could result in better classification, yet it reduces computational performance and add complexity.
#     - Smaller batch size is an option for better generalization, however it will decrease the update stability in each step.
#     - More epochs won't help in this case, since Config 2 and Config 3 stopped already early.
#     - We can choose more epoch with smaller learning rate. It will converge better but slower. Adaptive learning rate would overcome the the concern about the pace.
# ___


