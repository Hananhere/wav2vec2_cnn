import os
import torch
import torchaudio
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import Resample
import torch.nn.functional as F
