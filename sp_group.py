import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from nst_vgg19 import NST_VGG19_AdaIN
from modelscope import AutoModelForImageSegmentation

import sys
# Hack to fix a changed import in torchvision 0.17+, which otherwise breaks
# basicsr; see https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        import torchvision.transforms.functional as functional
        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        pass  # shrug...

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import gdown
import os
import argparse
from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer, Compression
from PIL import Image
from sp_pack import pack_psd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Создание модели RetinexNet
class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super().__init__()
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3, padding=4, padding_mode='replicate')
        self.net1_convs = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU()
        )
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size, padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)
        feats0 = self.net1_conv0(input_img)
        featss = self.net1_convs(feats0)
        outs = self.net1_recon(featss)
        R = torch.sigmoid(outs[:, 0:3, :, :])
        L = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L

class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super().__init__()
        self.relu = nn.ReLU()
        self.net2_conv0_1 = nn.Conv2d(4, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.net2_deconv1_1 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_deconv1_2 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_deconv1_3 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_fusion = nn.Conv2d(channel * 3, channel, kernel_size=1, padding=1, padding_mode='replicate')
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=0)

    def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1)
        out0 = self.net2_conv0_1(input_img)
        out1 = self.relu(self.net2_conv1_1(out0))
        out2 = self.relu(self.net2_conv1_2(out1))
        out3 = self.relu(self.net2_conv1_3(out2))
        out3_up = torch.nn.functional.interpolate(out3, size=(out2.size()[2], out2.size()[3]))
        deconv1 = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))
        deconv1_up = torch.nn.functional.interpolate(deconv1, size=(out1.size()[2], out1.size()[3]))
        deconv2 = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        deconv2_up = torch.nn.functional.interpolate(deconv2, size=(out0.size()[2], out0.size()[3]))
        deconv3 = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))
        deconv1_rs = torch.nn.functional.interpolate(deconv1, size=(input_R.size()[2], input_R.size()[3]))
        deconv2_rs = torch.nn.functional.interpolate(deconv2, size=(input_R.size()[2], input_R.size()[3]))
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)
        feats_fus = self.net2_fusion(feats_all)
        output = self.net2_output(feats_fus)
        return output

class RetinexNetWrapper(nn.Module):
    def __init__(self, decom_net_path, relight_net_path):
        super().__init__()
        self.decom_net = DecomNet()
        self.relight_net = RelightNet()
        self.load_weights(decom_net_path, relight_net_path)

    def load_weights(self, decom_net_path, relight_net_path):
        self.decom_net.load_state_dict(torch.load(decom_net_path))
        self.relight_net.load_state_dict(torch.load(relight_net_path))
        self.decom_net.eval()
        self.relight_net.eval()

    def forward(self, input_low):
        R_low, I_low = self.decom_net(input_low)
        I_delta = self.relight_net(I_low, R_low)
        I_delta_3 = torch.cat([I_delta, I_delta, I_delta], dim=1)
        output_S = R_low * I_delta_3
        return output_S

def flat_lights(image_np, model):
    """
    Применяет RetinexNet к изображению.
    :param image_np: Numpy-массив изображения (H, W, C) в диапазоне [0, 255].
    :return: Улучшенное изображение в виде Numpy-массива (H, W, C) в диапазоне [0, 255].
    """

    def preprocess_image(image_np):
        image = image_np.astype("float32") / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return torch.tensor(image).float()

    def postprocess_image(output_tensor):
        output_tensor = output_tensor.squeeze(0)
        output_array = output_tensor.detach().cpu().numpy()
        output_array = np.transpose(output_array, (1, 2, 0))
        output_array = np.clip(output_array * 255.0, 0, 255).astype(np.uint8)
        return output_array

    input_tensor = preprocess_image(image_np).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    enhanced_image_np = postprocess_image(output_tensor)

    return enhanced_image_np

def extract_foreground_mask(image_np, model):
    """
    Извлекает маску переднего плана из изображения.
    
    Args:
        image_np (np.ndarray): Numpy-массив изображения (H, W, C) в диапазоне [0, 255].
    
    Returns:
        np.ndarray: Маска в виде Numpy-массива (H, W) в диапазоне [0, 255].
    """
    image_size = (1024, 1024)
    
    # Преобразование изображения в тензор PyTorch
    transform_image = transforms.Compose([
        transforms.ToTensor(),  # Преобразует в тензор и нормализует в [0, 1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Нормализация для модели
    ])
    
    # Масштабирование изображения до целевого размера
    resized_image_np = cv2.resize(image_np, image_size, interpolation=cv2.INTER_LANCZOS4)
    input_tensor = transform_image(resized_image_np).unsqueeze(0).to('cuda').half()

    # Получение предсказаний модели
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    
    # Преобразование предсказания в маску
    pred = preds[0].squeeze().numpy()  # Тензор -> Numpy-массив
    mask = (pred * 255).clip(0, 255).astype(np.uint8)  # Нормализация в [0, 255]
    
    # Масштабирование маски обратно к исходному размеру изображения
    original_height, original_width = image_np.shape[:2]
    mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_LANCZOS4)
    
    return mask

def get_max_size(w, h, max_w, max_h):
    scale_w = max_w / w
    scale_h = max_h / h
    scale = min(scale_w, scale_h, 1.0)
    return int(w * scale), int(h * scale)

def load_image(img_path, max_width=2048, max_height=2048):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Not found: {img_path}")

    try:
        # Always load with alpha so we can extract the mask if needed
        pil_image = Image.open(img_path).convert("RGBA")
        rgba = np.array(pil_image)  # Shape: (H, W, 4)

        # Extract RGB and alpha channel
        original = rgba[:, :, :3]  # This ensures RGB output (H, W, 3)
        alpha = rgba[:, :, 3]     # Alpha channel for mask

        # If any pixel is transparent → use alpha as mask
        if np.any(alpha < 255):
            mask = alpha
        else:
            mask = None

    except Exception as e:
        raise ValueError(f"Can not load: {img_path}") from e

    print(img_path)
    
    # Resize if necessary
    height, width = original.shape[:2]
    new_width, new_height = get_max_size(width, height, max_width, max_height)
    
    if width != new_width or height != new_height:
        original = cv2.resize(original, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        if mask is not None:
            mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    return original, mask

def apply_mask(image_np, mask_np):
    if mask_np.shape[:2] != image_np.shape[:2]:
        raise ValueError("Image and mask size must be the same.")
    
    # Если маска одноканальная (grayscale), добавляем канал
    if len(mask_np.shape) == 2:
        mask_np = np.expand_dims(mask_np, axis=2)  # (H, W) -> (H, W, 1)
    
    # Добавляем альфа-канал к изображению
    if image_np.shape[2] == 3:  # Если изображение RGB
        image_with_alpha = np.concatenate(
            [image_np, mask_np], axis=2
        ).astype(np.uint8)
    else:
        raise ValueError("Image must be RGB (H, W, 3).")
    
    return image_with_alpha

def add_max_size_layer(psd, max_width, max_height, source_image_np):
    for layer in psd:
        if layer.name == 'max_size':
            return

    source_image = Image.fromarray(source_image_np)

    img_width, img_height = source_image.size

    left = (img_width - max_width) // 2
    top = (img_height - max_height) // 2
    right = left + max_width
    bottom = top + max_height

    cropped_image = source_image.crop((left, top, right, bottom))

    root = psd
    while root.parent:
        root = root.parent

    new_layer = PixelLayer.frompil(cropped_image, root, 'max_size', 0, 0, Compression.RAW)
    psd.append(new_layer)

def process_image(img_path, max_width, max_height, nst, psd, retinexnet=None, birefnet=None, refiner=None):
    try:
        original, mask = load_image(img_path, max_width * 2, max_height * 2)
    except Exception as e:
        return
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    corrected = flat_lights(original, retinexnet)

    if nst is not None:
        height, width = original.shape[:2]
        corrected, _ = refiner.enhance(corrected)
        corrected = nst(corrected)
        corrected = cv2.resize(corrected, (width, height), interpolation=cv2.INTER_LANCZOS4)

    if mask is None:
        mask = extract_foreground_mask(original, birefnet)

    corrected_rgba = apply_mask(corrected, mask)

    image = Image.fromarray(corrected_rgba, mode="RGBA")
    height, width = corrected.shape[:2]
    image = image.resize((width // 2, height // 2), Image.LANCZOS)

    bbox = image.getbbox()  # Получаем ограничивающую рамку
    if bbox is not None:  # Если есть непрозрачные пиксели
        cropped_image = image.crop(bbox)  # Обрезаем изображение
    else:
        print("^can not find foreground")
        return

    root = psd
    while root.parent:
        root = root.parent

    layer = PixelLayer.frompil(image, root, base_name, 0, 0, Compression.RAW)
    psd.append(layer)

def main():
    model_dir = os.path.expanduser('~/.cache/sprite-pipeline/')
    realesrgan_path = os.path.join(model_dir,'models/RealESRGAN_x2plus_mtg_v1.pth')
    decom_path = os.path.join(model_dir, 'models/decom.tar')
    relight_path = os.path.join(model_dir, 'models/relight.tar')
    if not (
        os.path.isfile(realesrgan_path) or
        os.path.isfile(decom_path) or
        os.path.isfile(relight_path)
    ):
        drive_path = 'https://drive.google.com/drive/folders/1gxAukn_M7YNbnWfg_OrV6BxlNWUuDPmL'
        gdown.download_folder(url=drive_path, output=model_dir, quiet=False, use_cookies=False)

    parser = argparse.ArgumentParser(description="Make PSD group from folder with sprites.")
    parser.add_argument("folder", help="Path to folder with images.")
    parser.add_argument("-s", "--style", required=False, help="Path to the style image.")
    parser.add_argument("-W", "--max_width", type=int, default=512, help="Max sprite width.")
    parser.add_argument("-H", "--max_height", type=int, default=512, help="Max sprite height.")
    parser.add_argument("-f", "--nst_force", type=float, default=1, help="Neural style transfer - weights mul.")
    parser.add_argument("-o", "--output", default="output.psd", help="Output PSD name.")
    args = parser.parse_args()

    esrgan2plus = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)

    upsampler = RealESRGANer(
        scale=1,
        model_path=realesrgan_path,
        dni_weight=None,
        model=esrgan2plus,
        tile=0,
        tile_pad=0,
        pre_pad=0,
        half=False,
        gpu_id=None)

    nst = None
    if args.style:
        style_image, _ = load_image(args.style)
        #style_image, _ = upsampler.enhance(style_image)
        nst = NST_VGG19_AdaIN(style_image, args.nst_force)

    retinexnet = RetinexNetWrapper(decom_path, relight_path).to(device)

    birefnet = AutoModelForImageSegmentation.from_pretrained('modelscope/BiRefNet', trust_remote_code=True)
    torch.set_float32_matmul_precision('high')
    birefnet.to(device)
    birefnet.eval()
    birefnet.half()

    try:
        psd_main = PSDImage.open(args.output)
    except Exception as e:
        psd_main = PSDImage.new(mode='RGBA', size=(1000, 1000))

    group_name = os.path.basename(os.path.normpath(args.folder))
    if args.style:
        group_name += '_' + os.path.basename(os.path.normpath(args.style))
        group_name += '_' + str(args.nst_force)
    group = Group.new(group_name, open_folder=False, parent=psd_main)
    for filename in os.listdir(args.folder):
        image_path = os.path.join(args.folder, filename)
        process_image(image_path, args.max_width, args.max_height, nst, group, retinexnet, birefnet, upsampler)

    add_max_size_layer(group, args.max_width, args.max_height, style_image)

    print('packing...')
    pack_psd(group)

    print('saving...')
    psd_main.save(args.output)

if __name__ == "__main__":
    main()