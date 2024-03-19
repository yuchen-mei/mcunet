import os
from tqdm import tqdm
import json

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data.distributed
from torchvision import datasets, transforms

from mcunet.model_zoo import build_model
from mcunet.utils import AverageMeter, accuracy, count_net_flops, count_parameters

import numpy as np
from bn_folder import bn_folding_model

# Training settings
parser = argparse.ArgumentParser()
# net setting
parser.add_argument('--net_id', type=str, help='net id of the model')
# data loader setting
parser.add_argument('--dataset', default='imagenet', type=str, choices=['imagenet', 'vww'])
parser.add_argument('--data-dir', default=os.path.expanduser('/dataset/imagenet/val'),
                    help='path to ImageNet validation data')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
# direct bfloat16 quantization setting
parser.add_argument('--bfloat16', action='store_true',
                    help='use bfloat16 quantization for weight and input if set, default is False')
# batch norm fusion setting
parser.add_argument('--fuse-bn', action='store_true',
                    help='use batch norm fusion, default is False')

args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = 'cuda'


def build_val_data_loader(resolution):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    kwargs = {'num_workers': args.workers, 'pin_memory': True}

    if args.dataset == 'imagenet':
        val_transform = transforms.Compose([
            transforms.Resize(int(resolution * 256 / 224)),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize
        ])
    elif args.dataset == 'vww':
        val_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),  # if center crop, the person might be excluded
            transforms.ToTensor(),
            normalize
        ])
    else:
        raise NotImplementedError
    val_dataset = datasets.ImageFolder(args.data_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    return val_loader


def validate(model, val_loader):
    model.eval()
    val_loss = AverageMeter()
    val_top1 = AverageMeter()
    if args.dataset=='imagenet': val_top5 = AverageMeter()

    with tqdm(total=len(val_loader), desc='Validate') as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.bfloat16:
                    data, target = data.to(device).bfloat16(), target.to(device)
                else:
                    data, target = data.to(device), target.to(device)

                output = model(data)
                val_loss.update(F.cross_entropy(output, target).item())
                if args.dataset=='vww':
                    top1 = accuracy(output, target, topk=(1,))[0]
                elif args.dataset=='imagenet':
                    top1, top5 = accuracy(output, target, topk=(1, 5))
                val_top1.update(top1.item(), n=data.shape[0])
                if args.dataset=='imagenet': val_top5.update(top5.item(), n=data.shape[0])
                if args.dataset=='imagenet':
                    t.set_postfix({'loss': val_loss.avg,
                                'top1': val_top1.avg,
                                'top5': val_top5.avg})
                elif args.dataset=='vww':
                    t.set_postfix({'loss': val_loss.avg,
                                'top1': val_top1.avg})
                t.update(1)

    if args.dataset=='imagenet': return val_top1.avg, val_top5.avg
    elif args.dataset=='vww': return val_top1.avg

def save_model_weights_biases(model, directory='weights_biases_dump'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for name, param in model.named_parameters():
        layer_type = 'weights' if 'weight' in name else 'biases'
        # Detach the tensor and convert to float32 before converting to numpy
        detached_param = param.detach().cpu()
        if detached_param.dtype == torch.bfloat16:
            detached_param = detached_param.to(torch.float32)
        numpy_param = detached_param.numpy()
        file_path = os.path.join(directory, f"{name.replace('.', '_')}_{layer_type}.npy")
        np.save(file_path, numpy_param)
    print(f"Model weights and biases have been saved in {directory}")

def save_activation(name):
    def hook(model, input, output):
        # Ensure directory exists
        directory = os.path.join('activations_dump', name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Convert BFloat16 to Float32 if necessary, then to numpy
        if output.dtype == torch.bfloat16:
            output_data = output.detach().to(torch.float32).cpu().numpy()
        else:
            output_data = output.detach().cpu().numpy()
        file_path = os.path.join(directory, f"{name}.npy")
        np.save(file_path, output_data)
    return hook


def register_activation_hooks(model):
    for name, layer in model.named_modules():
        # This checks if the layer is of any interest (e.g., not ReLU/Pooling since they might not have meaningful weights)
        if isinstance(layer, torch.nn.modules.conv.Conv2d) or isinstance(layer, torch.nn.modules.linear.Linear):
            layer.register_forward_hook(save_activation(name.replace('.', '_')))

def main():
    model, resolution, description = build_model(args.net_id, pretrained=True)

    if args.bfloat16:
        model = model.to(device).bfloat16()
    else:
        model = model.to(device)

    model.eval()

    if args.fuse_bn:
        model = bn_folding_model(model)
        # only dump activations for each layer with fused bn
        register_activation_hooks(model)
        # only dump weights/biases with fused bn
        save_model_weights_biases(model)

    val_loader = build_val_data_loader(resolution)

    # profile model
    total_macs = count_net_flops(model, [1, 3, resolution, resolution])
    total_params = count_parameters(model)
    print(' * FLOPs: {:.4}M, param: {:.4}M'.format(total_macs / 1e6, total_params / 1e6))

    if args.bfloat16: print('\033[92mUsing bfloat16\033[0m')
    if args.dataset=='imagenet':
        acc_top1, acc_top5 = validate(model, val_loader)
        print(' * Top-1 Accuracy: {:.2f}%   Top-5 Accuracy: {:.2f}%'.format(acc_top1, acc_top5))
    elif args.dataset=='vww':
        acc_top1 = validate(model, val_loader)
        print(' * Top-1 Accuracy: {:.2f}%'.format(acc_top1))


if __name__ == '__main__':
    main()
