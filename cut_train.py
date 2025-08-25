#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse, torch, random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

from models.resnet import ResNet18
from utils_lfba import CelebA_attr

# -------------------- hard-coded paths --------------------
trim_train_dir     = "/mnt/sdb/trimmed_train"                   # trimmed poisoned train
trim_train_glob    = "poisoned_batch_*.pt"

rand_src_dir       = "/home/suser/project/Thesis/saved_dataset" # full poisoned train (for random 2/3)
rand_src_glob      = "poisoned_batch_*.pt"

poison_test_dir    = "./saved_dataset"                          # poisoned test
poison_test_glob   = "poisoned_test_batch_*.pt"

celeba_root        = "/home/suser/project/Thesis/data"          # real CelebA root (clean test)
num_classes        = 8
device             = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size         = 256
epochs             = 50
lr                 = 0.01
momentum           = 0.9
weight_decay       = 5e-4
num_workers        = 8
seed               = 1234

torch.backends.cudnn.benchmark = True
torch.manual_seed(seed); random.seed(seed)

# -------------------- helpers --------------------
def load_pt_batches(folder, pattern):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    assert files, f"No files found: {folder}/{pattern}"
    xs, ys = [], []
    for f in files:
        obj = torch.load(f, map_location="cpu")
        xs.append(obj["data"]); ys.append(obj["label"])
    return torch.cat(xs, 0), torch.cat(ys, 0)

class TensorSubset(Dataset):
    def __init__(self, X, Y, indices):
        self.X, self.Y, self.indices = X, Y, indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i].item() if torch.is_tensor(self.indices) else self.indices[i]
        return self.X[idx], self.Y[idx]

def build_clean_celeba_loader(root, batch_size):
    tfm = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    ns = argparse.Namespace(data_dir=root)
    ds = CelebA_attr(ns, split="test", transforms=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, persistent_workers=True)

@torch.no_grad()
def eval_loader(model, loader):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total   += yb.size(0)
    return correct / max(1, total)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

# -------------------- data --------------------
# Trimmed train
X_trim, Y_trim = load_pt_batches(trim_train_dir, trim_train_glob)
trim_loader = DataLoader(TensorDataset(X_trim, Y_trim),
                         batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True, persistent_workers=True)
print(f"[INFO] trimmed train: {len(X_trim)}")

# Random 2/3 subset from full poisoned train
X_full, Y_full = load_pt_batches(rand_src_dir, rand_src_glob)
N = len(X_full); m = int(round(2 * N / 3))
perm = torch.randperm(N)
sel  = perm[:m]
rand_subset_loader = DataLoader(TensorSubset(X_full, Y_full, sel),
                                batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, persistent_workers=True)
print(f"[INFO] random 2/3 subset: {m} of {N}")

# Clean CelebA test
clean_loader = build_clean_celeba_loader(celeba_root, batch_size)
print("[INFO] built clean CelebA test loader.")

# Poisoned test
X_pois, Y_pois = load_pt_batches(poison_test_dir, poison_test_glob)
poison_loader = DataLoader(TensorDataset(X_pois, Y_pois),
                           batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True, persistent_workers=True)
print(f"[INFO] poisoned test: {len(X_pois)}")

# -------------------- models & optims --------------------
model_rand  = ResNet18(num_classes=num_classes).to(device)
model_trim  = ResNet18(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
opt_rand  = optim.SGD(model_rand.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
opt_trim  = optim.SGD(model_trim.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

# -------------------- train & compare --------------------
for epoch in range(1, epochs + 1):
    train_one_epoch(model_rand, rand_subset_loader, criterion, opt_rand)
    train_one_epoch(model_trim, trim_loader,        criterion, opt_trim)

    clean_rand = eval_loader(model_rand, clean_loader)
    asr_rand   = eval_loader(model_rand, poison_loader)

    clean_trim = eval_loader(model_trim, clean_loader)
    asr_trim   = eval_loader(model_trim, poison_loader)

    # exact format you asked for:
    print(f"epoche{epoch}: random clean={clean_rand:.4f} asr={asr_rand:.4f} , "
          f"trimmed clean={clean_trim:.4f} asr={asr_trim:.4f}")
