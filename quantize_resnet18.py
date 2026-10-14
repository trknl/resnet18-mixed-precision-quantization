import os
from pdb import run
import torch
import torch.nn as nn
import torchvision.models as models
import warnings
warnings.filterwarnings("ignore", message=".*fast_hadamard_transform.*")
warnings.filterwarnings("ignore", message=".*Named tensors.*")
import brevitas.nn as qnn
import torch.nn.utils.fusion as fusion
import copy
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.resnet import BasicBlock
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloatMSE
from brevitas.graph.calibrate import calibration_mode

# Weight quantizer used for all conv / linear layers built by make_quant_conv /
# make_quant_linear. Switch between the two to compare:
#   Int8WeightPerChannelFloat     -> per-channel scale from max |w|  (min-max)
#   Int8WeightPerChannelFloatMSE  -> per-channel scale that minimizes weight quant. MSE
WEIGHT_QUANT = Int8WeightPerChannelFloatMSE




# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = ".\data"
CHECKPOINT_PATH = r"C:\Users\MONSTER\Desktop\Quantization\Workflow Step 2\importing models\Transfer Learning\resnet18_cifar10_unfrozen_layer4_fc_best.pth"
NUM_CLASSES = 10
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



SAVE_PATH = r"C:\Users\MONSTER\Desktop\Quantization\quantized_resnet18.pth"
ONNX_PATH = r"C:\Users\MONSTER\Desktop\Quantization\quantized_resnet18.onnx"

# ---------------------------------------------------------------------------
# Data: CIFAR-10, resized to 224x224, normalized with ImageNet stats
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Defining the transform object: sequency of transforms to be applied to data in the main loop
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=16),   # mild augmentation
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# test_transform skips augmentation as there is no need
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Downloading the data if necessary, and then extracting the data to the CPU
# RAM as a numpy array
train_set = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=True, download=True, transform=train_transform
)
test_set = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=False, download=True, transform=test_transform
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

# Calibration data: TRAIN images but with the test transform (no random crop/flip),
# shuffled so each calibration run sees a random subset. Never use the test set here.
calib_set = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=True, download=True, transform=test_transform
)
calib_loader = DataLoader(calib_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)





# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_trained_model(checkpoint_path: str) -> nn.Module:
  model = models.resnet18(weights=None)
  model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
  checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
  model.load_state_dict(checkpoint["model_state_dict"])
  model.to(DEVICE)
  model.eval()
  return model

# ---------------------------------------------------------------------------
# 1. BatchNorm Fusion Model Wrapper
# ---------------------------------------------------------------------------
class fusedModel(nn.Module):
    def __init__(self, orig_model):
        super().__init__()
        self.model = copy.deepcopy(orig_model)
        self._fuse_layers(self.model)

    def _fuse_layers(self, module):
        children = list(module.named_children())

        for i, (name, child) in enumerate(children):
              if isinstance(child, nn.Conv2d):
                # Check if the NEXT child is a BatchNorm2d (common in ResNet blocks)
                if i + 1 < len(children) and isinstance(
                    children[i + 1][1], nn.BatchNorm2d
                ):
                  bn_name, bn_child = children[i + 1]

                  # Fuse them using PyTorch's utility
                  fused_conv = fusion.fuse_conv_bn_eval(child, bn_child)

                  # Replace Conv with the fused version
                  setattr(module, name, fused_conv)
                  # Replace BatchNorm with Identity so it does nothing
                  setattr(module, bn_name, nn.Identity())
              else:
                # Recurse deeper into containers (like BasicBlock or Sequential)
                self._fuse_layers(child)

    def forward(self, x):
            return self.model(x)  

# ---------------------------------------------------------------------------
# 2. Brevitas Quantized ResNet-18 Wrapper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ResNet-18 Direct Coordinate Indexing Reference Map:
# Root Children:
#   [0] conv1 (Conv2d)      [4] layer1 (Sequential)   [8] avgpool
#   [1] bn1 (BatchNorm2d)   [5] layer2 (Sequential)   [9] fc (Linear)
#   [2] relu (ReLU)         [6] layer3 (Sequential)
#   [3] maxpool             [7] layer4 (Sequential)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers to build Brevitas layers from float layers (per-channel weights)
# ---------------------------------------------------------------------------
def make_quant_conv(child: nn.Conv2d, weight_bit_width):
    q_conv = qnn.QuantConv2d(
        in_channels=child.in_channels,
        out_channels=child.out_channels,
        kernel_size=child.kernel_size,
        stride=child.stride,
        padding=child.padding,
        dilation=child.dilation,
        groups=child.groups,
        bias=(child.bias is not None),
        weight_quant=WEIGHT_QUANT,   # per-channel weight quantization (see WEIGHT_QUANT at top)
        weight_bit_width=weight_bit_width,
    )
    with torch.no_grad():
        q_conv.weight.copy_(child.weight)
        if child.bias is not None:
            q_conv.bias.copy_(child.bias)
    return q_conv


def make_quant_linear(child: nn.Linear, weight_bit_width):
    q_linear = qnn.QuantLinear(
        in_features=child.in_features,
        out_features=child.out_features,
        bias=(child.bias is not None),
        weight_quant=WEIGHT_QUANT,   # per-channel weight quantization (see WEIGHT_QUANT at top)
        weight_bit_width=weight_bit_width,
    )
    with torch.no_grad():
        q_linear.weight.copy_(child.weight)
        if child.bias is not None:
            q_linear.bias.copy_(child.bias)
    return q_linear


# ---------------------------------------------------------------------------
# BasicBlock replacement: TWO separate ReLUs + quantized residual add
# ---------------------------------------------------------------------------
class QuantBasicBlock(nn.Module):
    """
    Replacement for torchvision's BasicBlock (BN already fused into the convs).

    - relu1 / relu2 are separate QuantReLU modules, so each has its own scale.
    - conv2's output (and the downsample output, if present) pass through a
      QuantIdentity before the add, so both operands of the residual add are
      quantized. Without a downsample, `identity` is the block input, which is
      already the quantized output of the previous ReLU.
    - return_quant_tensor=False everywhere: no QuantTensor/float mixing in the add.
    """
    def __init__(self, conv1, conv2, downsample, act_bit_width):
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.downsample = downsample

        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width,
                                   return_quant_tensor=False,
                                   signed=False)
        self.conv2_out_quant = qnn.QuantIdentity(bit_width=act_bit_width,
                                                 return_quant_tensor=False)
        self.relu2 = qnn.QuantReLU(bit_width=act_bit_width,
                                   return_quant_tensor=False,
                                   signed=False)

        if downsample is not None:
            self.down_out_quant = qnn.QuantIdentity(bit_width=act_bit_width,
                                                    return_quant_tensor=False)
        else:
            self.down_out_quant = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.conv2_out_quant(out)

        if self.downsample is not None:
            identity = self.down_out_quant(self.downsample(x))

        out = out + identity      # both operands are quantized now
        return self.relu2(out)


# ---------------------------------------------------------------------------
# Single-layer quantization (same settings as the full model)
# ---------------------------------------------------------------------------
class FlexBasicBlock(nn.Module):
    """
    FLOAT BasicBlock rebuilt with the same structure as QuantBasicBlock:
    separate relu1 / relu2, plus Identity placeholders where QuantBasicBlock has
    its quantizers before the residual add. Numerically identical to the original
    block, but every stage is its own module, so any ONE of them can be swapped
    for a quantized version without touching the others.
    """
    def __init__(self, block):
        super().__init__()
        self.conv1 = block.conv1
        self.conv2 = block.conv2
        self.downsample = block.downsample
        self.relu1 = nn.ReLU()
        self.conv2_out_quant = nn.Identity()
        self.relu2 = nn.ReLU()
        self.down_out_quant = nn.Identity() if block.downsample is not None else None

    def forward(self, x):
        identity = x
        out = self.relu1(self.conv1(x))
        out = self.conv2_out_quant(self.conv2(out))
        if self.downsample is not None:
            identity = self.down_out_quant(self.downsample(x))
        return self.relu2(out + identity)


class QuantizeSingleLayer(nn.Module):
    """
    Quantizes exactly ONE module of a BN-fused ResNet-18 and leaves the rest float.

    Root children indices (after fusion):
      [0] conv1  [1] bn1(Identity)  [2] relu  [3] maxpool
      [4] layer1 [5] layer2 [6] layer3 [7] layer4  [8] avgpool  [9] fc

    Targets
    -------
    Top level (block_idx in 0, 2, 9):  conv1 / relu / fc
    Inside a block (block_idx 4-7, block_item_idx 0 or 1), layer_name is one of:
      "conv1", "conv2"   -> weights quantized (per-channel, WEIGHT_QUANT)
      "downsample"       -> the 1x1 conv of the skip path (layer2/3/4 block 0 only)
      "relu1", "relu2"   -> that ReLU alone becomes a QuantReLU (own scale)
      "conv2_out_quant"  -> quantizes conv2's output before the residual add
      "down_out_quant"   -> quantizes the skip-path output before the add

    Activation quantizers need calibration (calibrate_model) before evaluating.
    """
    BLOCK_LAYER_IDX = (4, 5, 6, 7)

    def __init__(self, orig_model, block_idx: int, block_item_idx: int = None,
                 layer_name: str = "conv1", weight_bit_width=8, act_bit_width=None):
        super().__init__()
        if act_bit_width is None:
            act_bit_width = weight_bit_width

        self.model = copy.deepcopy(orig_model)
        # Unwrap the inner ResNet if it is wrapped inside fusedModel
        target_net = self.model.model if hasattr(self.model, "model") else self.model

        root_names = list(dict(target_net.named_children()).keys())
        container_name = root_names[block_idx]
        container = getattr(target_net, container_name)

        if block_idx in self.BLOCK_LAYER_IDX:
            if block_item_idx is None:
                raise ValueError("block_item_idx is required for layer1-layer4")
            # Swap the block for its float, split-ReLU twin (same maths)
            split_block = FlexBasicBlock(container[block_item_idx])
            container[block_item_idx] = split_block

            if layer_name == "downsample":
                if split_block.downsample is None:
                    raise ValueError("This block has no downsample")
                parent_module, target_key = split_block.downsample, "0"
            else:
                parent_module, target_key = split_block, layer_name
        else:
            parent_module, target_key = target_net, container_name

        child = getattr(parent_module, target_key, None)
        if child is None:
            raise ValueError(f"Layer '{target_key}' not found (block has no such module)")

        print(f"Targeting -> block_idx={block_idx}, item={block_item_idx}, layer={layer_name}")
        print(child)

        if isinstance(child, nn.Conv2d):
            new_module = make_quant_conv(child, weight_bit_width)
        elif isinstance(child, nn.Linear):
            new_module = make_quant_linear(child, weight_bit_width)
        elif isinstance(child, nn.ReLU):
            new_module = qnn.QuantReLU(bit_width=act_bit_width,
                                       return_quant_tensor=False,
                                       signed=False)
        elif target_key in ("conv2_out_quant", "down_out_quant"):
            new_module = qnn.QuantIdentity(bit_width=act_bit_width,
                                           return_quant_tensor=False)
        else:
            raise ValueError(f"Don't know how to quantize '{target_key}' ({type(child).__name__})")

        setattr(parent_module, target_key, new_module)

    def forward(self, x):
        return self.model(x)


class QuantizedResNet18(nn.Module):
  """
  weight_bit_width : int  -> same precision for every conv / linear layer
                     list -> one entry per conv / linear layer (21 for ResNet-18),
                             in execution order (stem conv1, layer1.0.conv1, ..., fc)
  act_bit_width    : int, uniform for all activation quantizers
  After construction, self.layer_bits = [(layer_name, bits), ...]
  """

  def __init__(self, orig_model, weight_bit_width=8, act_bit_width=None):
    super(QuantizedResNet18, self).__init__()
    if act_bit_width is None:
      act_bit_width = weight_bit_width if isinstance(weight_bit_width, int) else 8

    self._w_spec = weight_bit_width
    self._w_idx = 0
    self.layer_bits = []

    self.model = copy.deepcopy(orig_model)
    self._quantize_layers(self.model, act_bit_width)

    if not isinstance(weight_bit_width, int) and self._w_idx != len(weight_bit_width):
      raise ValueError(f"The network has {self._w_idx} conv/linear layers but "
                       f"{len(weight_bit_width)} bit-widths were given.")

  def _next_weight_bits(self, label):
    if isinstance(self._w_spec, int):
      bits = self._w_spec
    else:
      if self._w_idx >= len(self._w_spec):
        raise ValueError(f"Too few bit-widths: {len(self._w_spec)} given, "
                         f"but layer '{label}' needs another one.")
      bits = self._w_spec[self._w_idx]
    self._w_idx += 1
    clean = label[len("model."):] if label.startswith("model.") else label
    self.layer_bits.append((clean, bits))
    return bits

  def _quantize_basic_block(self, block, act_bit_width, label):
    conv1 = make_quant_conv(block.conv1, self._next_weight_bits(f"{label}.conv1"))
    conv2 = make_quant_conv(block.conv2, self._next_weight_bits(f"{label}.conv2"))

    downsample = None
    if block.downsample is not None:
      mods = []
      for i, m in enumerate(block.downsample):
        if isinstance(m, nn.Conv2d):
          bits = self._next_weight_bits(f"{label}.downsample.{i}")
          mods.append(make_quant_conv(m, bits))
        else:
          mods.append(m)
      downsample = nn.Sequential(*mods)

    return QuantBasicBlock(conv1, conv2, downsample, act_bit_width)

  def _quantize_layers(self, module, act_bit_width, prefix=""):
    for name, child in module.named_children():
        label = prefix + name

        if isinstance(child, BasicBlock):
            setattr(module, name, self._quantize_basic_block(child, act_bit_width, label))

        elif isinstance(child, nn.Conv2d):
            setattr(module, name, make_quant_conv(child, self._next_weight_bits(label)))

        elif isinstance(child, nn.Linear):
            setattr(module, name, make_quant_linear(child, self._next_weight_bits(label)))

        elif isinstance(child, nn.ReLU):
            setattr(module, name, qnn.QuantReLU(bit_width=act_bit_width,
                                                return_quant_tensor=False,
                                                signed=False))
        else:
            self._quantize_layers(child, act_bit_width, prefix=label + ".")

  def forward(self, x):
    return self.model(x)

# ---------------------------------------------------------------------------
# Calibration Function
# ---------------------------------------------------------------------------
def calibrate_model(model, data_loader, num_batches=10):
  """Passes a few batches through the model to collect activation ranges."""
  model.eval()
  print("Starting calibration...")

  # calibration_mode: quantization is bypassed during these forward passes while the
  # activation quantizers collect their statistics. When the block exits, the scales
  # are frozen and quantization is switched back on.
  with torch.no_grad(), calibration_mode(model):
    for i, (images, _) in enumerate(data_loader):
      if i >= num_batches:
        break
      images = images.to(DEVICE)
      _ = model(images)

  print(f"Calibration completed using {num_batches} batches.")

  # ---------------------------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------------------------
def evaluate_model(model, data_loader):
  model.eval()
  correct = 0
  total = 0
  print("Evaluating model accuracy...")
  with torch.no_grad():
    for images, labels in data_loader:
      images, labels = images.to(DEVICE), labels.to(DEVICE)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total
  print(f"Quantized Model Accuracy: {accuracy:.2f}%")
  return accuracy

# ---------------------------------------------------------------------------
# Single-layer experiment: build -> evaluate -> calibrate -> evaluate
# ---------------------------------------------------------------------------
def run_single_layer(fused_model, block_idx, block_item_idx=None, layer_name="conv1",
                     weight_bits=4, act_bits=8, calib_batches=20):
    q = QuantizeSingleLayer(fused_model, block_idx, block_item_idx, layer_name,
                            weight_bit_width=weight_bits, act_bit_width=act_bits)
    q.to(DEVICE).eval()
    """
    print("Single layer, BEFORE calibration:")
    before = evaluate_model(q, test_loader)
    """
    calibrate_model(q, calib_loader, num_batches=calib_batches)

    print("Single layer, AFTER calibration:")
    after = evaluate_model(q, test_loader)
    #return before, after
    return after


# ---------------------------------------------------------------------------
# Save / Load quantized model
# ---------------------------------------------------------------------------
def save_quantized_model(q_model, path, weight_bits, act_bits, accuracy=None):
    """Saves weights + calibrated scales + the info needed to rebuild the model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": q_model.state_dict(),   # includes calibrated activation scales
        "weight_bit_width": weight_bits,            # list of 21 ints
        "act_bit_width": act_bits,
        "layer_bits": q_model.layer_bits,           # (layer_name, bits) for reference
        "weight_quantizer": WEIGHT_QUANT.__name__,
        "num_classes": NUM_CLASSES,
        "accuracy": accuracy,
    }, path)
    print(f"Quantized model saved to: {path}")


def load_quantized_model(path, device=DEVICE):
    """Rebuilds the quantized architecture and loads the saved weights/scales."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Skeleton float model (its weights get overwritten by load_state_dict)
    base = models.resnet18(weights=None)
    base.fc = nn.Linear(base.fc.in_features, ckpt["num_classes"])
    base.eval()
    fused = fusedModel(base)

    q_model = QuantizedResNet18(fused,
                                weight_bit_width=ckpt["weight_bit_width"],
                                act_bit_width=ckpt["act_bit_width"])
    q_model.load_state_dict(ckpt["model_state_dict"])
    q_model.to(device).eval()
    return q_model


def export_quantized_onnx(q_model, path, input_size=(1, 3, 224, 224)):
    """Optional: export to ONNX (QCDQ format) for use outside PyTorch."""
    from brevitas.export import export_onnx_qcdq
    q_model = q_model.cpu().eval()
    dummy = torch.randn(*input_size)
    export_onnx_qcdq(q_model, args=dummy, export_path=path, opset_version=13)
    print(f"ONNX model saved to: {path}")

# ---------------------------------------------------------------------------
# Main Execution Flow
# --------------------------------------------------------------------------
def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"File not found: {CHECKPOINT_PATH}")
        return

    # Load pre-trained float model
    model = load_trained_model(CHECKPOINT_PATH)
    
    # 1. Instantiate fused model (bakes BatchNorm stats into Convolutions)
    fused_model = fusedModel(model)
    fused_model.eval()

    """
    # obtained 77.77% accuracy
    figure11_LAYERS = [
    # (block_idx, block_item_idx, layer_name, bits)
    (0, None, "conv1",      8),   #  1 stem conv1
    (4, 0,    "conv1",      8),   #  2 layer1.0.conv1
    (4, 0,    "conv2",      5),   #  3 layer1.0.conv2
    (4, 1,    "conv1",      6),   #  4 layer1.1.conv1
    (4, 1,    "conv2",      5),   #  5 layer1.1.conv2
    (5, 0,    "conv1",      5),   #  6 layer2.0.conv1
    (5, 0,    "conv2",      5),   #  7 layer2.0.conv2
    (5, 0,    "downsample", 8),   #  8 layer2.0.downsample
    (5, 1,    "conv1",      5),   #  9 layer2.1.conv1
    (5, 1,    "conv2",      4),   # 10 layer2.1.conv2
    (6, 0,    "conv1",      4),   # 11 layer3.0.conv1
    (6, 0,    "conv2",      3),   # 12 layer3.0.conv2
    (6, 0,    "downsample", 3),   # 13 layer3.0.downsample
    (6, 1,    "conv1",      4),   # 14 layer3.1.conv1
    (6, 1,    "conv2",      3),   # 15 layer3.1.conv2
    (7, 0,    "conv1",      3),   # 16 layer4.0.conv1
    (7, 0,    "conv2",      3),   # 17 layer4.0.conv2
    (7, 0,    "downsample", 2),   # 18 layer4.0.downsample
    (7, 1,    "conv1",      2),   # 19 layer4.1.conv1
    (7, 1,    "conv2",      2),   # 20 layer4.1.conv2
    (9, None, "fc",         8),   # 21 fc
    ]
"""
    """
    # obtained 13.72% accuracy
    figure12_LAYERS = [
        # (block_idx, block_item_idx, layer_name, bits)
        (0, None, "conv1",      8),   #  1 stem conv1
        (4, 0,    "conv1",      3),   #  2 layer1.0.conv1
        (4, 0,    "conv2",      3),   #  3 layer1.0.conv2
        (4, 1,    "conv1",      3),   #  4 layer1.1.conv1
        (4, 1,    "conv2",      2),   #  5 layer1.1.conv2
        (5, 0,    "conv1",      3),   #  6 layer2.0.conv1
        (5, 0,    "conv2",      3),   #  7 layer2.0.conv2
        (5, 0,    "downsample", 4),   #  8 layer2.0.downsample
        (5, 1,    "conv1",      3),   #  9 layer2.1.conv1
        (5, 1,    "conv2",      2),   # 10 layer2.1.conv2
        (6, 0,    "conv1",      3),   # 11 layer3.0.conv1
        (6, 0,    "conv2",      3),   # 12 layer3.0.conv2
        (6, 0,    "downsample", 2),   # 13 layer3.0.downsample
        (6, 1,    "conv1",      3),   # 14 layer3.1.conv1
        (6, 1,    "conv2",      2),   # 15 layer3.1.conv2
        (7, 0,    "conv1",      3),   # 16 layer4.0.conv1
        (7, 0,    "conv2",      3),   # 17 layer4.0.conv2
        (7, 0,    "downsample", 2),   # 18 layer4.0.downsample
        (7, 1,    "conv1",      3),   # 19 layer4.1.conv1
        (7, 1,    "conv2",      2),   # 20 layer4.1.conv2
        (9, None, "fc",         8),   # 21 fc
    ]
    """

    """
    # obtained 94.22% accuracy
    # about 5.0 bits average
    LAYERS = [
    # (block_idx, block_item_idx, layer_name, bits)
    (0, None, "conv1",      8),   #  1 stem
    (4, 0,    "conv1",      8),   #  2
    (4, 0,    "conv2",      8),   #  3
    (4, 1,    "conv1",      8),   #  4
    (4, 1,    "conv2",      8),   #  5
    (5, 0,    "conv1",      8),   #  6
    (5, 0,    "conv2",      8),   #  7
    (5, 0,    "downsample", 8),   #  8
    (5, 1,    "conv1",      8),   #  9
    (5, 1,    "conv2",      8),   # 10
    (6, 0,    "conv1",      8),   # 11
    (6, 0,    "conv2",      8),   # 12
    (6, 0,    "downsample", 8),   # 13
    (6, 1,    "conv1",      8),   # 14
    (6, 1,    "conv2",      8),   # 15
    (7, 0,    "conv1",      4),   # 16
    (7, 0,    "conv2",      4),   # 17
    (7, 0,    "downsample", 8),   # 18
    (7, 1,    "conv1",      4),   # 19
    (7, 1,    "conv2",      4),   # 20
    (9, None, "fc",         8),   # 21
]
    """

    """
    # obtained 92.6% accuracy
    # about 4.2 bits average
    LAYERS = [
    # (block_idx, block_item_idx, layer_name, bits)
    (0, None, "conv1",      8),   #  1 stem
    (4, 0,    "conv1",      8),   #  2
    (4, 0,    "conv2",      8),   #  3
    (4, 1,    "conv1",      8),   #  4
    (4, 1,    "conv2",      8),   #  5
    (5, 0,    "conv1",      6),   #  6
    (5, 0,    "conv2",      6),   #  7
    (5, 0,    "downsample", 8),   #  8
    (5, 1,    "conv1",      6),   #  9
    (5, 1,    "conv2",      6),   # 10
    (6, 0,    "conv1",      4),   # 11
    (6, 0,    "conv2",      4),   # 12
    (6, 0,    "downsample", 8),   # 13
    (6, 1,    "conv1",      4),   # 14
    (6, 1,    "conv2",      4),   # 15
    (7, 0,    "conv1",      4),   # 16
    (7, 0,    "conv2",      4),   # 17
    (7, 0,    "downsample", 8),   # 18
    (7, 1,    "conv1",      4),   # 19
    (7, 1,    "conv2",      4),   # 20
    (9, None, "fc",         8),   # 21
]
"""

    LAYERS = [
    # (block_idx, block_item_idx, layer_name, bits)
    (0, None, "conv1",      8),   #  1 stem
    (4, 0,    "conv1",      8),   #  2
    (4, 0,    "conv2",      8),   #  3
    (4, 1,    "conv1",      8),   #  4
    (4, 1,    "conv2",      8),   #  5
    (5, 0,    "conv1",      8),   #  6
    (5, 0,    "conv2",      8),   #  7
    (5, 0,    "downsample", 8),   #  8
    (5, 1,    "conv1",      8),   #  9
    (5, 1,    "conv2",      8),   # 10
    (6, 0,    "conv1",      8),   # 11
    (6, 0,    "conv2",      8),   # 12
    (6, 0,    "downsample", 8),   # 13
    (6, 1,    "conv1",      8),   # 14
    (6, 1,    "conv2",      8),   # 15
    (7, 0,    "conv1",      8),   # 16
    (7, 0,    "conv2",      8),   # 17
    (7, 0,    "downsample", 8),   # 18
    (7, 1,    "conv1",      8),   # 19
    (7, 1,    "conv2",      8),   # 20
    (9, None, "fc",         4),   # 21
]

    bits = [l[3] for l in LAYERS]          # the 21 bit-widths from the table

    q_model = QuantizedResNet18(fused_model,
                            weight_bit_width=bits,    # list of 21
                            act_bit_width=8)
    q_model.to(DEVICE).eval()

    calibrate_model(q_model, calib_loader, num_batches=20)   # once, at the end
    acc = evaluate_model(q_model, test_loader)    

    # Save for use elsewhere
    save_quantized_model(q_model, SAVE_PATH, weight_bits=bits, act_bits=8, accuracy=acc)
    
if __name__ == "__main__":
    main()