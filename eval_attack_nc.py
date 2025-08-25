#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import time
import argparse
import random
from types import SimpleNamespace

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------- Optional (for saving images) ----------------------
try:
    import torchvision
    import torchvision.utils as vutils
    HAS_TV = True
except Exception:
    torchvision = None
    vutils = None
    HAS_TV = False

# ---------------------- Local models (required) ----------------------
from models.resnet import ResNet18
from models.vgg import vgg16_bn, vgg11_bn
from models.preact_resnet import PreActResNet18
from models.cnn_mnist import CNN_MNIST
from models.googlenet import GoogLeNet

# ---------------------- Local dataset helpers (required) ----------------------
from utils_lfba import load_dataset, load_data


# ============================== B 风格：用户设置 ==============================
# 仅 --ckpt 走 CLI，其余参数在此处修改（与脚本 B 一致的使用体验）
model_name        = "resnet"      # resnet | vgg11 | vgg16 | preact_resnet | cnn_mnist | googlenet
num_classes       = 8
device            = "cuda:0"
batch_size        = 256
max_samples       = 2048  # 仅用于 B 的采样上限（A 内部不强制使用）

dataset_name      = "celeba"      # 传入 utils_lfba.load_dataset
data_dir          = "./data"
num_workers       = 2

# A 风格：Neural Cleanse (原脚本 A 的关键超参)
# nc_epoch          = 10
nc_lr             = 0.01
atk_succ_threshold= 99.0
early_stop        = True
early_stop_threshold = 99.0
early_stop_patience  = 25

patience          = 5
cost_multiplier   = 2.0
init_cost         = 1e-3

EPSILON           = 1e-7
n_times_test      = 1 
attack_mode       = "clean"  
poison_ratio      = 0.1        
surrogate_model   = "vgg16"    

# 输出根目录（B 风格）
root_nc_dir       = "nc/results"

# 全局：供 manifest 使用
ckpt_path = None
nc_epoch = 5              # was 10
# atk_succ_threshold = 100.0  # was 99.0
# patience = 10             # was 5
# cost_multiplier = 1.2     # was 2.0
# init_cost = 5e-4          # was 1e-

# ============================== CLI ==============================

def parse_args():
    parser = argparse.ArgumentParser(description="Neural Cleanse")
    parser.add_argument("--ckpt", type=str, required=True, help="(.pt)")
    return parser.parse_args()


# ============================== B 风格：工具函数 ==============================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _infer_shape_from_loader(loader: DataLoader):
    x0, _ = next(iter(loader))
    return x0.shape  # (B,C,H,W)


def get_model_from_settings() -> torch.nn.Module:
    dev = torch.device(device)
    mdl = model_name.lower()

    if mdl == "resnet":
        model = ResNet18(num_classes=num_classes)
    elif mdl == "vgg16":
        model = vgg16_bn(num_classes=num_classes)
    elif mdl == "vgg11":
        model = vgg11_bn(num_classes=num_classes)
    elif mdl == "preact_resnet":
        model = PreActResNet18(num_classes=num_classes)
    elif mdl == "cnn_mnist":
        model = CNN_MNIST()
    elif mdl == "googlenet":
        model = GoogLeNet()
    else:
        raise ValueError(f"Unknown model: {mdl}")

    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt)
    return model.to(dev).eval()


def build_clean_loader_local(max_samples_limit: int = None) -> DataLoader:
    args_ds = SimpleNamespace(data=dataset_name, data_dir=data_dir, batch_size=batch_size)
    trainset, testset = load_dataset(args_ds)
    _, test_loader = load_data(args_ds, trainset, testset)

    if max_samples_limit is None:
        return test_loader

    # Cap 到 max_samples
    xs, ys, n = [], [], 0
    for x, y in test_loader:
        xs.append(x); ys.append(y); n += x.size(0)
        if n >= max_samples_limit:
            break
    ds = TensorDataset(torch.cat(xs,0), torch.cat(ys,0))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


class Normalize:
    def __init__(self, args, expected_values, variance):
        self.n_channels = args.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, args, expected_values, variance):
        self.n_channels = args.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


class RegressionModel(nn.Module):
    def __init__(self, args, init_mask, init_pattern):
        self._EPSILON = args.EPSILON
        super(RegressionModel, self).__init__()
        self.mask_tanh = nn.Parameter(torch.tensor(init_mask))
        self.pattern_tanh = nn.Parameter(torch.tensor(init_pattern))

        self.classifier = self._get_classifier(args)  # 最小改动：从 args 取预加载模型
        self.normalizer = self._get_normalize(args)
        self.denormalizer = self._get_denormalize(args)

    def forward(self, x):
        mask = self.get_raw_mask()
        pattern = self.get_raw_pattern()
        if self.normalizer:
            pattern = self.normalizer(self.get_raw_pattern())
        x = (1 - mask) * x + mask * pattern
        return self.classifier(x)

    def get_raw_mask(self):
        mask = nn.Tanh()(self.mask_tanh)
        return mask / (2 + self._EPSILON) + 0.5

    def get_raw_pattern(self):
        pattern = nn.Tanh()(self.pattern_tanh)
        return pattern / (2 + self._EPSILON) + 0.5

    def _get_classifier(self, args):

        if hasattr(args, 'preloaded_classifier') and args.preloaded_classifier is not None:
            classifier = args.preloaded_classifier
        else:
            # 以下保留 A 的构造分支（通常不会走到）
            if args.data == "mnist":
                classifier = CNN_MNIST()
            elif args.data == "cifar10":
                classifier = PreActResNet18(num_classes=10)
            elif args.data == "gtsrb":
                classifier = PreActResNet18(num_classes=43)
            elif args.data == "imagenet":
                classifier = ResNet18(num_classes=200)
            elif args.data == "celeba":
                classifier = ResNet18(num_classes=8)
            elif args.data == 'svhn':
                classifier = CNN_MNIST()
            else:
                raise Exception("Invalid Dataset")
        for param in classifier.parameters():
            param.requires_grad = False
        classifier.eval()
        return classifier.to(args.device)

    def _get_denormalize(self, args):
        if args.data == "cifar10":
            denormalizer = Denormalize(args, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif args.data == "mnist":
            denormalizer = Denormalize(args, [0.5], [0.5])
        elif args.data == "gtsrb":
            denormalizer = None
        elif args.data == "imagenet":
            denormalizer = None
        elif args.data == "celeba":
            denormalizer = None
        elif args.data == 'svhn':
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, args):
        if args.data == "cifar10":
            normalizer = Normalize(args, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif args.data == "mnist":
            normalizer = Normalize(args, [0.5], [0.5])
        elif args.data == "gtsrb":
            normalizer = None
        elif args.data == "imagenet":
            normalizer = None
        elif args.data == "celeba":
            normalizer = None
        elif args.data == 'svhn':
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        return normalizer


class Recorder:
    def __init__(self, args):
        super().__init__()

        # Best optimization results
        self.mask_best = None
        self.pattern_best = None
        self.reg_best = float("inf")

        # Logs and counters for adjusting balance cost
        self.logs = []
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # Counter for early stop
        self.early_stop_counter = 0
        self.early_stop_reg_best = self.reg_best

        # Cost
        self.cost = args.init_cost
        self.cost_multiplier_up = args.cost_multiplier
        self.cost_multiplier_down = args.cost_multiplier ** 1.5

    def reset_state(self, args):
        self.cost = args.init_cost
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False
        print("Initialize cost to {:f}".format(self.cost))

    def save_result_to_dir(self, args):
        result_dir = os.path.join(args.result, args.data)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_dir = os.path.join(result_dir, args.attack_mode)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_dir = os.path.join(result_dir, str(args.target_label))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        pattern_best = self.pattern_best
        mask_best = self.mask_best
        trigger = pattern_best * mask_best

        path_mask = os.path.join(result_dir, "mask.png")
        path_pattern = os.path.join(result_dir, "pattern.png")
        path_trigger = os.path.join(result_dir, "trigger.png")

        if torchvision is not None:
            torchvision.utils.save_image(mask_best, path_mask, normalize=True)
            torchvision.utils.save_image(pattern_best, path_pattern, normalize=True)
            torchvision.utils.save_image(trigger, path_trigger, normalize=True)


def train(args, init_mask, init_pattern):
    train_dataset, test_dataset = load_dataset(args)
    _, test_loader = load_data(args, train_dataset, test_dataset)

    regression_model = RegressionModel(args, init_mask, init_pattern).to(args.device)

    optimizerR = torch.optim.Adam(regression_model.parameters(), lr=args.lr, betas=(0.5, 0.9))

    recorder = Recorder(args)

    for epoch in range(args.epoch):
        early_stop = train_step(regression_model, optimizerR, test_loader, recorder, epoch, args)
        if early_stop:
            break

    recorder.save_result_to_dir(args)

    return recorder, args


def train_step(regression_model, optimizerR, dataloader, recorder, epoch, args):
    print("Epoch {} - Label: {} | {} - {}:".format(epoch, args.target_label, args.data, args.attack_mode))
    cross_entropy = nn.CrossEntropyLoss()
    total_pred = 0
    true_pred = 0

    loss_ce_list = []
    loss_reg_list = []
    loss_list = []
    loss_acc_list = []

    inner_early_stop_flag = False
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        optimizerR.zero_grad()

        inputs = inputs.to(args.device)
        sample_num = inputs.shape[0]
        total_pred += sample_num
        target_labels = torch.ones((sample_num), dtype=torch.int64).to(args.device) * args.target_label
        predictions = regression_model(inputs)

        loss_ce = cross_entropy(predictions, target_labels)
        loss_reg = torch.norm(regression_model.get_raw_mask(), 2)
        total_loss = loss_ce + recorder.cost * loss_reg
        total_loss.backward()
        optimizerR.step()

        minibatch_accuracy = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() * 100.0 / sample_num
        loss_ce_list.append(loss_ce.detach())
        loss_reg_list.append(loss_reg.detach())
        loss_list.append(total_loss.detach())
        loss_acc_list.append(minibatch_accuracy)

        true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()

    loss_ce_list = torch.stack(loss_ce_list)
    loss_reg_list = torch.stack(loss_reg_list)
    loss_list = torch.stack(loss_list)
    loss_acc_list = torch.stack(loss_acc_list)

    avg_loss_ce = torch.mean(loss_ce_list)
    avg_loss_reg = torch.mean(loss_reg_list)
    avg_loss = torch.mean(loss_list)
    avg_loss_acc = torch.mean(loss_acc_list)

    if avg_loss_acc >= args.atk_succ_threshold and avg_loss_reg < recorder.reg_best:
        recorder.mask_best = regression_model.get_raw_mask().detach()
        recorder.pattern_best = regression_model.get_raw_pattern().detach()
        recorder.reg_best = avg_loss_reg
        recorder.save_result_to_dir(args)
        print(" Updated !!!")

    print(
        "  Result: Accuracy: {:.3f} | Cross Entropy Loss: {:.6f} | Reg Loss: {:.6f} | Reg best: {:.6f}".format(
            true_pred * 100.0 / total_pred, avg_loss_ce, avg_loss_reg, recorder.reg_best
        )
    )

    if args.early_stop:
        if recorder.reg_best < float("inf"):
            if recorder.reg_best >= args.early_stop_threshold * recorder.early_stop_reg_best:
                recorder.early_stop_counter += 1
            else:
                recorder.early_stop_counter = 0

        recorder.early_stop_reg_best = min(recorder.early_stop_reg_best, recorder.reg_best)

        if (
            recorder.cost_down_flag
            and recorder.cost_up_flag
            and recorder.early_stop_counter >= args.early_stop_patience
        ):
            print("Early_stop !!!")
            inner_early_stop_flag = True

    if not inner_early_stop_flag:
        if recorder.cost == 0 and avg_loss_acc >= args.atk_succ_threshold:
            recorder.cost_set_counter += 1
            if recorder.cost_set_counter >= args.patience:
                recorder.reset_state(args)
        else:
            recorder.cost_set_counter = 0

        if avg_loss_acc >= args.atk_succ_threshold:
            recorder.cost_up_counter += 1
            recorder.cost_down_counter = 0
        else:
            recorder.cost_up_counter = 0
            recorder.cost_down_counter += 1

        if recorder.cost_up_counter >= args.patience:
            recorder.cost_up_counter = 0
            print("Up cost from {} to {}".format(recorder.cost, recorder.cost * recorder.cost_multiplier_up))
            recorder.cost *= recorder.cost_multiplier_up
            recorder.cost_up_flag = True

        elif recorder.cost_down_counter >= args.patience:
            recorder.cost_down_counter = 0
            print("Down cost from {} to {}".format(recorder.cost, recorder.cost / recorder.cost_multiplier_down))
            recorder.cost /= recorder.cost_multiplier_down
            recorder.cost_down_flag = True

        if recorder.mask_best is None:
            recorder.mask_best = regression_model.get_raw_mask().detach()
            recorder.pattern_best = regression_model.get_raw_pattern().detach()

    return inner_early_stop_flag


# ============================== MAD + 保存（B 风格） ==============================

def tv_norm(mask: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    dh = mask[:, :, 1:, :] - mask[:, :, :-1, :]
    dw = mask[:, :, :, 1:] - mask[:, :, :, :-1]
    return (dh.abs().pow(beta).mean() + dw.abs().pow(beta).mean())


def anomaly_index_mad_from_results(results):
    classes = sorted(results.keys())
    # 使用 A 脚本的 L1 口径：sum(|mask|)
    l1s = np.array([float(torch.sum(torch.abs(results[c]['mask']))) for c in classes], dtype=np.float64)
    med = np.median(l1s)
    mad = np.median(np.abs(l1s - med))
    ai = np.abs(l1s - med) / (mad + 1e-6)
    return [(int(c), float(l1s[i]), float(ai[i])) for i, c in enumerate(classes)]


def save_nc_artifacts(results, out_dir: str, image_shape, config=None):
    os.makedirs(out_dir, exist_ok=True)

    rows = [("class","mask_l1","mask_tv","final_ce")]
    all_triggers = {}

    for c, obj in results.items():
        cls_dir = os.path.join(out_dir, f"class_{c}")
        os.makedirs(cls_dir, exist_ok=True)

        torch.save({'mask': obj['mask'], 'pattern': obj['pattern']}, os.path.join(cls_dir, "trigger.pt"))


        if HAS_TV and vutils is not None:
            try:
                vutils.save_image(obj['pattern'], os.path.join(cls_dir, "pattern.png"))
                m = obj['mask']
                if m.dim() == 3:
                    m = m.unsqueeze(0)
                vutils.save_image(m.repeat(1,3,1,1), os.path.join(cls_dir, "mask.png"))
                vutils.save_image(obj['pattern'] * m, os.path.join(cls_dir, "trigger.png"))
            except Exception:
                pass

        rows.append((c, obj['mask_l1'], obj['mask_tv'], obj.get('final_ce', 0.0)))
        all_triggers[c] = {'mask': obj['mask'], 'pattern': obj['pattern']}

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerows(rows)

    torch.save(all_triggers, os.path.join(out_dir, "all_triggers.pt"))

    mad = anomaly_index_mad_from_results(results)
    with open(os.path.join(out_dir, "mad.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class","mask_l1","AI"])
        for (c,l1,ai) in mad:
            w.writerow([c,l1,ai])

    top = max(mad, key=lambda x: x[2]) if len(mad) else (None, None, None)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ckpt_path": ckpt_path,
        "model_name": model_name,
        "num_classes": num_classes,
        "dataset_name": dataset_name,
        "data_dir": data_dir,
        "image_shape": {"C": int(image_shape[0]), "H": int(image_shape[1]), "W": int(image_shape[2])},
        "a_core_hyperparams": {
            "epoch": nc_epoch,
            "lr": nc_lr,
            "init_cost": init_cost,
            "atk_succ_threshold": atk_succ_threshold,
            "early_stop": early_stop,
            "early_stop_threshold": early_stop_threshold,
            "early_stop_patience": early_stop_patience,
            "patience": patience,
            "cost_multiplier": cost_multiplier,
        },
        "mad_top_suspicious": {
            "class": int(top[0]) if top[0] is not None else None,
            "ai": float(top[2]) if top[2] is not None else None
        },
        "per_class": [
            {"class": int(c), "mask_l1": float(l1), "ai": float(ai)}
            for (c,l1,ai) in mad
        ]
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


# ============================== main ==========================================

def main():
    global ckpt_path
    args_cli = parse_args()
    ckpt_path = args_cli.ckpt

    set_seed(0)


    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    nc_dir = os.path.join(root_nc_dir, ckpt_name)

    model = get_model_from_settings()

    argsA = SimpleNamespace(
        # A 期望字段
        log_dir='./logs/',
        data_dir=data_dir,
        model=model_name,
        data=dataset_name,
        device=device,
        attack_mode=attack_mode,
        result=nc_dir,            
        model_dir="./saved_model/", 
        poison_ratio=poison_ratio,
        n_objs=2,
        surrogate_model=surrogate_model,
        batch_size=batch_size,
        lr=nc_lr,
        input_height=None,
        input_width=None,
        input_channel=None,
        init_cost=init_cost,
        atk_succ_threshold=atk_succ_threshold,
        early_stop=early_stop,
        early_stop_threshold=early_stop_threshold,
        early_stop_patience=early_stop_patience,
        patience=patience,
        cost_multiplier=cost_multiplier,
        epoch=nc_epoch,
        num_workers=num_workers,
        target_label=0,
        total_label=None,
        EPSILON=EPSILON,
        to_file=True,
        n_times_test=n_times_test,
        # 自定义扩展：用于 _get_classifier 直接返回
        preloaded_classifier=model,
    )


    if argsA.data == "mnist" or argsA.data == "cifar10":
        argsA.total_label = 10
    elif argsA.data == "gtsrb":
        argsA.total_label = 43
    elif argsA.data == "imagenet":
        argsA.total_label = 200
    elif argsA.data == "celeba":
        argsA.total_label = 8
    elif argsA.data == 'svhn':
        argsA.total_label = 10
    else:
        raise Exception("Invalid Dataset")

    if argsA.data == "cifar10":
        argsA.input_height = 32; argsA.input_width = 32; argsA.input_channel = 3
    elif argsA.data == "gtsrb":
        argsA.input_height = 32; argsA.input_width = 32; argsA.input_channel = 3
    elif argsA.data == "mnist":
        argsA.input_height = 28; argsA.input_width = 28; argsA.input_channel = 1
    elif argsA.data == "imagenet":
        argsA.input_height = 64; argsA.input_width = 64; argsA.input_channel = 3
    elif argsA.data == "celeba":
        argsA.input_height = 64; argsA.input_width = 64; argsA.input_channel = 3
    elif argsA.data == "svhn":
        argsA.input_height = 32; argsA.input_width = 32; argsA.input_channel = 3
    else:
        raise Exception("Invalid Dataset")


    init_mask = np.ones((1, argsA.input_height, argsA.input_width)).astype(np.float32)
    init_pattern = np.ones((argsA.input_channel, argsA.input_height, argsA.input_width)).astype(np.float32)

    results_b_style = {}

    for test_idx in range(argsA.n_times_test):
        print(f"Test {test_idx}:")

        for target_label in range(argsA.total_label):
            print(f"----------------- Analyzing label: {target_label} -----------------")
            argsA.target_label = target_label
            recorder, _ = train(argsA, init_mask, init_pattern)

            mask = recorder.mask_best.detach().cpu().float()
            pattern = recorder.pattern_best.detach().cpu().float()

            # 计算 L1 与 TV（B 风格函数），A 的 mask 形状是 (1,H,W)
            m4 = mask.unsqueeze(0) if mask.dim() == 3 else mask
            m_l1 = float(torch.sum(torch.abs(mask)))  # 与 A 的 MAD 口径保持一致（sum |m|）
            m_tv = float(tv_norm(m4))

            results_b_style[target_label] = {
                'mask': m4,           # 存 4D，便于统一处理
                'pattern': pattern.unsqueeze(0) if pattern.dim() == 3 else pattern,
                'mask_l1': m_l1,
                'mask_tv': m_tv,
                'final_ce': 0.0,
            }


    mad_table = anomaly_index_mad_from_results(results_b_style)
    print("\n=== Neural Cleanse (MAD, A) results ===")
    for c, l1, ai in mad_table:
        print(f"Class {c}: L1(mask)={l1:.6f}, AI={ai:.2f}")
    if mad_table:
        top = max(mad_table, key=lambda x: x[2])
        print(f"Most suspicious class: {top[0]} (AI={top[2]:.2f})")


    clean_loader = build_clean_loader_local(max_samples)
    _, C, H, W = _infer_shape_from_loader(clean_loader)
    save_nc_artifacts(results_b_style, nc_dir, (C,H,W), config=None)

    print(f"\nSaved NC artifacts to: {os.path.abspath(nc_dir)}")


if __name__ == "__main__":
    main()
