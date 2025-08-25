# run_strip_batch.py
import os, glob, cv2, argparse, numpy as np, torch
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

# project imports
from models.resnet import ResNet18
from utils_lfba import load_dataset, load_data

# =========================
# Your provided config
# =========================
CHECKPOINTS = OrderedDict([
    # ("blend",   "/mnt/sdb/models/train_attack_blend_resnet_celeba_0.1_blend_no_smooth_epoch48.pt"),
    # ("sig",     "/mnt/sdb/models/train_attack_sig_resnet_celeba_0.1_sig_no_smooth_epoch50.pt"),
    # ("square",  "/mnt/sdb/models/train_attack_square_resnet_celeba_0.1_square_no_smooth_epoch30.pt"),
    # ("ftrojan", "/mnt/sdb/models/train_attack_ftrojan_resnet_celeba_0.1_ftrojan_no_smooth_epoch44.pt"),
    # ("HCBsmile","/mnt/sdb/models/train_attack_HCBsmile_resnet_celeba_0.1_HCBsmile_no_smooth_epoch3.pt"),
    # ("HCB","/mnt/sdb/models/train_attack_HCB_resnet_celeba_0.1_HCB_no_smooth_epoch14.pt"),
])

# Optional: poisoned data table (not used by NC, kept here for future ASR checks)
POISON_TABLE = (
    # ("HCBsmile", "./saved_dataset",             "poisoned_test_batch_*.pt"),
    # ("blend",    "./saved_dataset",             "resnet_blend_poisoned_test_batch_*.pt"),
    # ("sig",      "./saved_dataset",             "resnet_sig_poisoned_test_batch_*.pt"),
    # ("square",   "/mnt/sdb/dataset_checkpoint", "resnet_square_poisoned_test_batch_*.pt"),
    # ("ftrojan",  "/mnt/sdb/dataset_checkpoint", "resnet_ftrojan_poisoned_test_batch_*.pt"),
    ("HCB",  "/mnt/sdb/dataset_checkpoint", "resnet_HCB_poisoned_test_batch_*.pt"),
)

# =========================
# (De)Normalization helpers
# =========================
def make_norm_transforms(mean, std):
    norm = transforms.Normalize(mean=mean, std=std)
    inv = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)],
                               std=[1.0/s for s in std])
    return norm, inv

def chw_to_hwc_uint8(img_t, inv_norm):
    x = inv_norm(img_t.clone()).clamp(0, 1).cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)

def hwc_uint8_to_chw_float(img_hwc_uint8):
    x = img_hwc_uint8.astype(np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))

# =========================
# STRIP (aligned)
# =========================
class STRIP:
    def __init__(self, classifier, overlay_dataset, n_sample, device, norm, inv_norm,
                 alpha=1.0, beta=1.0):
        self.clf = classifier.to(device).eval()
        self.dataset = overlay_dataset
        self.n_sample = n_sample
        self.device = device
        self.norm = norm
        self.inv = inv_norm
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def __call__(self, background_hwc_uint8):
        idxs = np.random.randint(0, len(self.dataset), size=self.n_sample)
        batch_list = []
        for idx in idxs:
            overlay_t, _ = self.dataset[idx]
            ov_hwc = chw_to_hwc_uint8(overlay_t, self.inv)
            mixed_hwc = cv2.addWeighted(background_hwc_uint8, self.alpha, ov_hwc, self.beta, 0)
            mixed_chw = hwc_uint8_to_chw_float(mixed_hwc)
            mixed_t = torch.from_numpy(mixed_chw)
            mixed_t = self.norm(mixed_t).to(self.device)
            batch_list.append(mixed_t)

        batch = torch.stack(batch_list, dim=0)
        logits = self.clf(batch)
        probs = F.softmax(logits, dim=1).clamp_min(1e-12)
        probs_np = probs.cpu().numpy()
        entropy_sum = -np.nansum(probs_np * np.log2(probs_np))
        return float(entropy_sum / self.n_sample)

# =========================
# Poisoned set loader
# =========================
def load_poisoned_tensor_dataset(poisoned_dir, pattern):
    files = sorted(glob.glob(os.path.join(poisoned_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No poisoned batches at {poisoned_dir} matching {pattern}")
    xs, ys = [], []
    for f in files:
        batch = torch.load(f, map_location='cpu')
        xs.append(batch['data'])
        ys.append(batch['label'])
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return TensorDataset(x, y)

def find_poison_info(ckpt_path):
    for key, pdir, patt in POISON_TABLE:
        if key in ckpt_path:
            return key, pdir, patt
    # fallback: try by filename
    base = os.path.basename(ckpt_path)
    for key, pdir, patt in POISON_TABLE:
        if key in base:
            return key, pdir, patt
    raise ValueError(f"No POISON_TABLE match for checkpoint: {ckpt_path}")

# =========================
# Core runner
# =========================
def run_once(ckpt_path, data_dir, batch_size, device, n_sample, n_test, start_point,
             mean, std, alpha, beta, target_label, outdir):
    os.makedirs(outdir, exist_ok=True)
    tag, poisoned_dir, poison_pattern = find_poison_info(ckpt_path)
    print(f"\n=== Running STRIP for [{tag}] ===")
    print(f"ckpt: {ckpt_path}")
    print(f"poisoned_dir/pattern: {poisoned_dir} / {poison_pattern}")

    device = torch.device(device)

    # Model
    model = ResNet18(num_classes=8)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt)
    model.to(device).eval()

    # Data (CelebA)
    ds_args = argparse.Namespace(
        data='celeba',
        data_dir=data_dir,
        batch_size=batch_size,
        num_classes=8,
    )
    trainset, testset = load_dataset(ds_args)
    train_loader, test_loader = load_data(ds_args, trainset, testset)
    test_loader = DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=True)

    # Poisoned set
    bd_dataset = load_poisoned_tensor_dataset(poisoned_dir, poison_pattern)
    bd_loader = DataLoader(bd_dataset, batch_size=batch_size, shuffle=True)

    # Norms
    norm, inv_norm = make_norm_transforms(mean, std)

    # STRIP object
    strip = STRIP(model, train_loader.dataset, n_sample=n_sample, device=device,
                  norm=norm, inv_norm=inv_norm, alpha=alpha, beta=beta)

    clean_ents, poison_ents = [], []

    # One paired batch (like the reference)
    for (clean_x, _), (bd_x, _) in zip(test_loader, bd_loader):
        clean_x = clean_x.to(device)
        bd_x = bd_x.to(device)

        # Quick ASR check (no overlays)
        with torch.no_grad():
            preds = model(bd_x).argmax(1).cpu()
        asr = (preds == target_label).float().mean().item()
        print(f"[Quick ASR on first poisoned batch] target={target_label}  ASR={asr:.2%}")

        end_idx = min(start_point + n_test, clean_x.size(0), bd_x.size(0))
        print(f"Running STRIP on {end_idx - start_point} samples from first batch...")
        for i in tqdm(range(start_point, end_idx)):
            bg_clean_hwc  = chw_to_hwc_uint8(clean_x[i].detach().cpu(), inv_norm)
            bg_poison_hwc = chw_to_hwc_uint8(bd_x[i].detach().cpu(),    inv_norm)
            clean_ents.append(strip(bg_clean_hwc))
            poison_ents.append(strip(bg_poison_hwc))
        break

    # Save + report
    clean_min, clean_max, clean_avg = float(np.min(clean_ents)), float(np.max(clean_ents)), float(np.mean(clean_ents))
    poison_min, poison_max, poison_avg = float(np.min(poison_ents)), float(np.max(poison_ents)), float(np.mean(poison_ents))

    npz_path = os.path.join(outdir, f"entropies_celeba_{tag}.npz")
    np.savez(npz_path, clean=clean_ents, poison=poison_ents)
    print(f"Saved entropies to: {npz_path}")
    print(f"Clean  : min={clean_min:.4f}, max={clean_max:.4f}, avg={clean_avg:.4f}")
    print(f"Poison : min={poison_min:.4f}, max={poison_max:.4f}, avg={poison_avg:.4f}")

def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch_size', type=int, default=500)
    ap.add_argument('--n_sample', type=int, default=100)
    ap.add_argument('--n_test', type=int, default=100)
    ap.add_argument('--start_point', type=int, default=0)

    # Norm (defaults to your CelebA stats from the snippet you pasted)
    ap.add_argument('--mean', type=float, nargs=3, default=[0.506, 0.425, 0.383])
    ap.add_argument('--std',  type=float, nargs=3, default=[0.271, 0.263, 0.276])

    # Blend weights (keep as 1/1 to mirror “should_be”; change to 0.7/0.3 if you want bg-dominant)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--beta',  type=float, default=1.0)

    ap.add_argument('--target_label', type=int, default=7)
    ap.add_argument('--outdir', default='strip_results')

    # Optional: run a subset, e.g. --only square sig
    ap.add_argument('--only', nargs='*', default=None,
                    help='subset of keys to run (blend, sig, square, ftrojan, HCBsmile)')
    return ap.parse_args()

def main():
    args = parse_cli()

    # choose which to run
    items = CHECKPOINTS.items()
    if args.only:
        keys = set(args.only)
        items = [(k, v) for k, v in CHECKPOINTS.items() if k in keys]

    for tag, ckpt_path in items:
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] missing checkpoint: {ckpt_path}")
            continue
        try:
            run_once(
                ckpt_path=ckpt_path,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                device=args.device,
                n_sample=args.n_sample,
                n_test=args.n_test,
                start_point=args.start_point,
                mean=args.mean,
                std=args.std,
                alpha=args.alpha,
                beta=args.beta,
                target_label=args.target_label,
                outdir=args.outdir,
            )
        except Exception as e:
            print(f"[ERROR] {tag}: {e}")

if __name__ == "__main__":
    main()
