# eval_attack_strip_aligned.py
import os
import glob
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

# your project stuff
from models.resnet import ResNet18
from utils_lfba import load_dataset, load_data

# ----------------------------
# Helpers: (de)normalization
# ----------------------------
def make_norm_transforms(mean, std):
    """
    Returns (norm, inv_norm) as torchvision.transforms.Normalize
    norm: x -> (x-mean)/std
    inv:  x -> x*std + mean
    """
    norm = transforms.Normalize(mean=mean, std=std)
    inv = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1.0/s for s in std],
    )
    return norm, inv

def chw_to_hwc_uint8(img_t, inv_norm):
    """
    img_t: torch Tensor (C,H,W) normalized
    inv -> pixel-ish space (0..1), then to uint8 HWC for OpenCV.
    """
    x = inv_norm(img_t.clone())
    x = x.clamp(0, 1)  # be safe
    x = x.cpu().numpy()
    x = np.transpose(x, (1, 2, 0))  # CHW -> HWC
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x

def hwc_uint8_to_chw_float(img_hwc_uint8):
    """
    OpenCV output (HWC, uint8) -> CHW float32 in [0,1]
    """
    x = img_hwc_uint8.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    return x

# ----------------------------
# STRIP core (aligned to "should_be_strip")
# ----------------------------
class STRIP:
    def __init__(self, classifier, overlay_dataset, n_sample, device, norm, inv_norm,
                 alpha=1.0, beta=1.0):
        """
        classifier: torch.nn.Module
        overlay_dataset: dataset to sample random overlays from (should be clean train set)
        n_sample: number of overlays per background
        device: 'cuda:0' or 'cpu'
        norm, inv_norm: torchvision Normalize transforms
        alpha,beta: cv2.addWeighted weights (kept 1.0/1.0 to match "should_be")
        """
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
        """
        background_hwc_uint8: ONE background image in pixel space (HWC, uint8).
        Returns mean entropy across n_sample overlays (like should_be).
        """
        # sample overlay indices
        idxs = np.random.randint(0, len(self.dataset), size=self.n_sample)

        batch_list = []
        for idx in idxs:
            overlay_t, _ = self.dataset[idx]  # (C,H,W) normalized tensor

            # inverse-normalize overlay to pixel HWC uint8
            ov_hwc = chw_to_hwc_uint8(overlay_t, self.inv)

            # pixel-space blend (OpenCV expects HWC, uint8)
            mixed_hwc = cv2.addWeighted(background_hwc_uint8, self.alpha, ov_hwc, self.beta, 0)

            # back to CHW float [0,1] -> normalize for the classifier
            mixed_chw = hwc_uint8_to_chw_float(mixed_hwc)
            mixed_t = torch.from_numpy(mixed_chw)
            mixed_t = self.norm(mixed_t).to(self.device)

            batch_list.append(mixed_t)

        batch = torch.stack(batch_list, dim=0)  # (n_sample, C, H, W)

        # logits -> softmax -> per-sample entropy; sum then / n_sample
        logits = self.clf(batch)
        probs = F.softmax(logits, dim=1).clamp_min(1e-12)  # avoid log(0)
        probs_np = probs.cpu().numpy()
        entropy_sum = -np.nansum(probs_np * np.log2(probs_np))
        return float(entropy_sum / self.n_sample)

# ----------------------------
# I/O for poisoned batches saved as .pt
# ----------------------------
def load_poisoned_tensor_dataset(poisoned_dir, pattern):
    files = sorted(glob.glob(os.path.join(poisoned_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No poisoned batches found at {poisoned_dir} with pattern {pattern}")
    imgs, labels = [], []
    for f in files:
        batch = torch.load(f, map_location='cpu')
        imgs.append(batch['data'])   # (N,C,H,W) normalized
        labels.append(batch['label'])
    x = torch.cat(imgs, dim=0)
    y = torch.cat(labels, dim=0)
    return TensorDataset(x, y)

# ----------------------------
# CLI + main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # fixed to ResNet + CelebA per your request
    p.add_argument('--ckpt_path', type=str, required=True)
    p.add_argument('--data_dir', type=str, default='./data')
    p.add_argument('--poisoned_dir', type=str, default='./saved_dataset')
    p.add_argument('--poison_format', type=str, default='poisoned_test_batch_*.pt')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--num_classes', type=int, default=8)

    # STRIP params
    p.add_argument('--n_sample', type=int, default=100, help='overlays per background')
    p.add_argument('--n_test', type=int, default=100, help='how many backgrounds from the first batch')
    p.add_argument('--start_point', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=500)

    # “should_be” style decision (optional)
    p.add_argument('--detection_boundary', type=float, default=0.2)

    # CelebA norm defaults (tweak if your pipeline uses different stats)
    p.add_argument('--mean', type=float, nargs=3, default=[0.506, 0.425, 0.383])
    p.add_argument('--std',  type=float, nargs=3, default=[0.271, 0.263, 0.276])

    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs('strip_results', exist_ok=True)

    device = torch.device(args.device)

    # ----------------------------
    # Model (ResNet18 only)
    # ----------------------------
    model = ResNet18(num_classes=args.num_classes)
    ckpt = torch.load(args.ckpt_path, map_location=device,weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt)
    model.to(device).eval()

    # ----------------------------
    # Data: clean train/test for overlays + clean backgrounds
    # ----------------------------
    # Use your utils to get CelebA loaders
    ds_args = argparse.Namespace(
        data='celeba',
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        # the rest are ignored by your utils if not needed
    )
    trainset, testset = load_dataset(ds_args)
    train_loader, test_loader = load_data(ds_args, trainset, testset)
    # make sure test loader is shuffled so the first batch is random (like should_be)
    test_loader = DataLoader(test_loader.dataset, batch_size=args.batch_size, shuffle=True)

    # ----------------------------
    # Data: poisoned set from saved .pt batches
    # ----------------------------
    bd_dataset = load_poisoned_tensor_dataset(args.poisoned_dir, args.poison_format)
    bd_loader = DataLoader(bd_dataset, batch_size=args.batch_size, shuffle=True)

    # ----------------------------
    # Norm / inv-norm
    # ----------------------------
    norm, inv_norm = make_norm_transforms(args.mean, args.std)

    # ----------------------------
    # STRIP object (aligned behavior)
    # ----------------------------
    # alpha=1.0, beta=1.0 to mimic should_be's cv2.addWeighted(background,1,overlay,1,0)
    strip = STRIP(model, train_loader.dataset, n_sample=args.n_sample,
                  device=device, norm=norm, inv_norm=inv_norm,
                  alpha=1, beta=1)

    clean_ents, poison_ents = [], []

    # ----------------------------
    # Evaluate only ONE paired batch (then break), like should_be
    # ----------------------------
    for (clean_x, clean_y), (bd_x, bd_y) in zip(test_loader, bd_loader):
        clean_x = clean_x.to(device)
        bd_x = bd_x.to(device)

        target_label = getattr(args, "target_label", 7)  # or hardcode 7 if you prefer
        model.eval()
        with torch.no_grad():
            preds = model(bd_x).argmax(1).cpu()
        asr = (preds == target_label).float().mean().item()
        print(f"[Quick ASR on first poisoned batch] target={target_label}  ASR={asr:.2%}")

        end_idx = min(args.start_point + args.n_test, clean_x.size(0), bd_x.size(0))

        print(f'Running STRIP on {end_idx - args.start_point} samples from first batch...')
        for i in tqdm(range(args.start_point, end_idx)):
            # backgrounds (pixel-space HWC uint8)
            bg_clean_hwc = chw_to_hwc_uint8(clean_x[i].detach().cpu(), inv_norm)
            bg_poison_hwc = chw_to_hwc_uint8(bd_x[i].detach().cpu(), inv_norm)

            # mean entropy over random overlays
            ent_clean = strip(bg_clean_hwc)
            ent_poison = strip(bg_poison_hwc)

            clean_ents.append(ent_clean)
            poison_ents.append(ent_poison)
        break  # only first batch

    # ----------------------------
    # Save + report
    # ----------------------------
    clean_min, clean_max, clean_avg = float(np.min(clean_ents)), float(np.max(clean_ents)), float(np.mean(clean_ents))
    poison_min, poison_max, poison_avg = float(np.min(poison_ents)), float(np.max(poison_ents)), float(np.mean(poison_ents))

    np.savez(f'strip_results/entropies_celeba_aligned.npz', clean=clean_ents, poison=poison_ents)

    print(f"\nClean entropy:  min={clean_min:.4f}, max={clean_max:.4f}, avg={clean_avg:.4f}")
    print(f"Poison entropy: min={poison_min:.4f}, max={poison_max:.4f}, avg={poison_avg:.4f}")

    # should_be-style crude decision on min entropy (optional)
    overall_min = min(clean_min, poison_min)
    # print(f"Min entropy trojan: {overall_min:.4f}, Detection boundary: {args.detection_boundary:.3f}")
    # if overall_min < args.detection_boundary:
    #     print("A backdoored model\n")
    # else:
    #     print("Not a backdoor model\n")

if __name__ == "__main__":
    main()


# python eval_attack_strip.py \
#   --ckpt_path /mnt/sdb/models/train_attack_square_resnet_celeba_0.1_square_no_smooth_epoch30.pt \
#   --data_dir ./data \
#   --poisoned_dir /mnt/sdb/dataset_checkpoint \
#   --poison_format "resnet_square_poisoned_test_batch_*.pt" \
#   --device cuda:0 \
#   --num_classes 8 \
#   --n_sample 100 \
#   --n_test 100 \
#   --batch_size 500 

# python eval_attack_strip.py \
#   --ckpt_path /mnt/sdb/models/train_attack_HCBsmile_resnet_celeba_0.1_HCBsmile_no_smooth_epoch3.pt \
#   --data_dir ./data \
#   --poisoned_dir /mnt/sdb/dataset_checkpoint \
#   --poison_format "poisoned_test_batch_*.pt" \
#   --device cuda:0 \
#   --num_classes 8 \
#   --n_sample 100 \
#   --n_test 100 \
#   --batch_size 500 
