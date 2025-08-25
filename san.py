# eval_norm_sanity.py
import os, glob, argparse, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms as T

from utils_lfba import load_dataset, load_data  # your utils

# ---------- transforms ----------
class Identity:
    def __call__(self, x): return x

def make_norm(mean, std):
    return T.Normalize(mean=mean, std=std)

def make_inv_norm(mean, std):
    # inverse of Normalize(mean, std)
    return T.Normalize(mean=[-m/s for m,s in zip(mean, std)],
                       std=[1.0/s for s in std])

def compose_data_to_model(data_mean, data_std, model_mean, model_std):
    """Map dataset output -> model expected input."""
    if all(abs(m) < 1e-9 for m in data_mean) and all(abs(s-1.0) < 1e-9 for s in data_std):
        return make_norm(model_mean, model_std)  # dataset is [0,1]
    else:
        return T.Compose([make_inv_norm(data_mean, data_std),
                          make_norm(model_mean, model_std)])

# ---------- dataset wrappers ----------
class PostNormDataset(Dataset):
    def __init__(self, base, post): self.base, self.post = base, post
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x,y = self.base[i]
        return self.post(x), y

class TensorDatasetWithTransform(Dataset):
    def __init__(self, x, y, post): self.x, self.y, self.post = x, y, post
    def __len__(self): return self.x.size(0)
    def __getitem__(self, i):
        return self.post(self.x[i]), self.y[i]

# ---------- poisoned loader ----------
def load_poisoned_tensor_dataset(poisoned_dir, pattern):
    files = sorted(glob.glob(os.path.join(poisoned_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No poisoned batches match {poisoned_dir}/{pattern}")
    xs, ys = [], []
    for f in files:
        batch = torch.load(f, map_location='cpu')
        xs.append(batch['data'])   # (N,C,H,W)
        ys.append(batch['label'])
    return torch.cat(xs, 0), torch.cat(ys, 0)

# ---------- model loader ----------
def load_model_resnet18(ckpt_path, num_classes, device):
    from models.resnet import ResNet18
    model = ResNet18(num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif isinstance(ckpt, dict):
        sd = ckpt
    elif isinstance(ckpt, torch.nn.Module):
        return ckpt.to(device).eval()
    else:
        raise ValueError("Unknown checkpoint format")

    # strip 'module.' if present
    if any(k.startswith('module.') for k in sd.keys()):
        sd = {k.replace('module.', '', 1): v for k,v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()

# ---------- eval helpers ----------
@torch.no_grad()
def eval_accuracy(model, loader, device, max_batches=None):
    model.eval()
    tot = corr = 0
    for bi, (x,y) in enumerate(loader):
        x = x.to(device); y = y.to(device)
        pred = model(x).argmax(1)
        corr += (pred == y).sum().item()
        tot  += y.numel()
        if max_batches is not None and bi+1 >= max_batches:
            break
    return corr / max(1, tot)

@torch.no_grad()
def eval_asr(model, loader, device, target_label, max_batches=None):
    model.eval()
    tot = hit = 0
    for bi, (x,_) in enumerate(loader):
        x = x.to(device)
        pred = model(x).argmax(1).cpu().numpy()
        hit += (pred == target_label).sum()
        tot += pred.size
        if max_batches is not None and bi+1 >= max_batches:
            break
    return hit / max(1, tot)

# ---------- main ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_path', required=True)
    ap.add_argument('--data_dir', default='./data')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--num_classes', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--data', default='celeba')

    # What the model was trained with:
    ap.add_argument('--model_mean', type=float, nargs=3, default=[0.5,0.5,0.5])
    ap.add_argument('--model_std',  type=float, nargs=3, default=[0.5,0.5,0.5])

    # What the dataset CURRENTLY outputs (often identity if only ToTensor())
    ap.add_argument('--data_mean', type=float, nargs=3, default=[0.0,0.0,0.0])
    ap.add_argument('--data_std',  type=float, nargs=3, default=[1.0,1.0,1.0])

    # Optional poisoned set to compute ASR
    ap.add_argument('--poisoned_dir', default=None)
    ap.add_argument('--poison_format', default="poisoned_test_batch_*.pt")
    ap.add_argument('--target_label', type=int, default=7)

    # Speed limiters
    ap.add_argument('--max_clean_batches', type=int, default=None)
    ap.add_argument('--max_poison_batches', type=int, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1) Load model
    model = load_model_resnet18(args.ckpt_path, args.num_classes, device)

    # 2) Load clean data
    # Reuse your utils
    args.data = args.data or 'celeba'
    trainset, testset = load_dataset(args)
    # Wrap test with post-normalization to match model training
    post = compose_data_to_model(args.data_mean, args.data_std, args.model_mean, args.model_std)
    test_wrapped = PostNormDataset(testset, post)
    test_loader = DataLoader(test_wrapped, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 3) Evaluate clean accuracy
    acc = eval_accuracy(model, test_loader, device, max_batches=args.max_clean_batches)
    print(f"[Clean accuracy] {acc:.2%}")

    # 4) Optional: evaluate ASR on poisoned set (no overlays)
    if args.poisoned_dir:
        px, py = load_poisoned_tensor_dataset(args.poisoned_dir, args.poison_format)
        poison_wrapped = TensorDatasetWithTransform(px, py, post)
        poison_loader = DataLoader(poison_wrapped, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers, pin_memory=True)
        asr = eval_asr(model, poison_loader, device, args.target_label, max_batches=args.max_poison_batches)
        print(f"[ASR on poisoned data] target={args.target_label}  ASR={asr:.2%}  (N={len(poison_wrapped)})")

if __name__ == "__main__":
    main()
