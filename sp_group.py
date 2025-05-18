import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
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

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print('running on', device)
torch.set_float32_matmul_precision('high')

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
        self.decom_net.load_state_dict(torch.load(decom_net_path, map_location=device))
        self.relight_net.load_state_dict(torch.load(relight_net_path, map_location=device))
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

def extract_foreground_mask_birefnet(image_np, model):
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
    input_tensor = transform_image(resized_image_np).unsqueeze(0).to(device).half()

    # Получение предсказаний модели
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    
    # Преобразование предсказания в маску
    pred = preds[0].squeeze().numpy()  # Тензор -> Numpy-массив
    mask = (pred * 255).clip(0, 255).astype(np.uint8)  # Нормализация в [0, 255]
    
    return mask

def get_max_size(w, h, max_w, max_h):
    scale_w = max_w / w
    scale_h = max_h / h
    scale = min(scale_w, scale_h, 1.0)
    return int(w * scale), int(h * scale)

def get_nonzero_bbox(mask, padding=1):
    """
    Возвращает координаты bounding box вокруг ненулевых пикселей в маске:
    (x_min, y_min, x_max, y_max)
    
    padding: количество пикселей для добавления с каждой стороны
    """
    if mask is None:
        return None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None  # Пустая маска

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Добавляем padding
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, mask.shape[1] - 1)
    y_max = min(y_max + padding, mask.shape[0] - 1)

    return x_min, y_min, x_max + 1, y_max + 1  # x1, y1, x2, y2

class NotAnImageException(Exception):
    pass

def extract_sprites(img_path, max_width=2048, max_height=2048, birefnet=None):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Not found: {img_path}")

    try:
        pil_image = Image.open(img_path).convert("RGBA")
    except Exception as e:
        raise NotAnImageException(f"Can not load: {img_path}") from e

    rgba = np.array(pil_image)  # Shape: (H, W, 4)
    original = rgba[:, :, :3]   # RGB
    alpha = rgba[:, :, 3]       # Alpha channel

    mask = None
    if np.any(alpha < 255):
        mask = alpha.copy()
    else:
        mask = extract_foreground_mask_birefnet(original, birefnet)
        if mask is None:
            print("^can not find foreground")
            mask = np.ones_like(alpha, dtype=np.uint8) * 255
        else:
            mask = cv2.resize(mask, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    # Создаем полное RGBA изображение с маской
    rgba_combined = rgba.copy()
    if mask is not None:
        rgba_combined[:, :, 3] = mask

    # Получаем размеры изображения
    height, width = rgba_combined.shape[:2]

    # Ищем связные компоненты по маске (альфа канал)
    alpha_mask = rgba_combined[:, :, 3]
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (alpha_mask > 0).astype(np.uint8), connectivity=8
    )

    sprites = []

    for stat in stats[1:]:  # Пропускаем фон (статистика индекса 0)
        x, y, w, h, area = stat
        if area < 10:  # Фильтруем слишком маленькие области
            continue

        if h < 4 or w < 4:  # Если сторона меньше 4 — пропускаем
            continue

        x1, y1 = x, y
        x2, y2 = x + w, y + h

        # Добавляем 1 пиксель границы
        padding = 1
        x1 = max(x1 - padding, 0)
        y1 = max(y1 - padding, 0)
        x2 = min(x2 + padding, width)
        y2 = min(y2 + padding, height)

        # Вырезаем спрайт (включая альфу)
        sprite_rgba = rgba_combined[y1:y2, x1:x2]

        # Ресайз под max размеры
        h_sprite, w_sprite = sprite_rgba.shape[:2]
        new_w, new_h = get_max_size(w_sprite, h_sprite, max_width, max_height)

        if w_sprite != new_w or h_sprite != new_h:
            sprite_rgba = cv2.resize(sprite_rgba, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        sprites.append(sprite_rgba)

    return sprites

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

def process_sprite(original, mask, nst, retinexnet=None, refiner=None):
    height, width = original.shape[:2]
    
    corrected = flat_lights(original, retinexnet)

    if nst is not None:
        corrected, _ = refiner.enhance(corrected)
        corrected = nst(corrected)
        corrected = cv2.resize(corrected, (width, height), interpolation=cv2.INTER_LANCZOS4)

    corrected_rgba = apply_mask(corrected, mask)

    image = Image.fromarray(corrected_rgba, mode="RGBA")
    height, width = corrected.shape[:2]
    image = image.resize((width // 2, height // 2), Image.LANCZOS)

    return image

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
    parser.add_argument("-W", "--max_width", type=int, default=480, help="Max sprite width.")
    parser.add_argument("-H", "--max_height", type=int, default=480, help="Max sprite height.")
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
        pil_image = Image.open(args.style).convert("RGB")
        style_image = np.array(pil_image)
        nst = NST_VGG19_AdaIN(style_image, args.nst_force)

    retinexnet = RetinexNetWrapper(decom_path, relight_path).to(device)

    if device == torch.device('cpu'):
        print('Warning: torch device is CPU so foreground extraction is ultra slow.')

    birefnet = AutoModelForImageSegmentation.from_pretrained('modelscope/BiRefNet', trust_remote_code=True)
    birefnet.to(device).eval().half()

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

        try:
            sprites = extract_sprites(image_path, args.max_width * 2, args.max_height * 2, birefnet)

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            if len(sprites) == 1:
                print(base_name)
            else:
                print(base_name + ': ' + str(len(sprites)))

            for i, rgba in enumerate(sprites):
                original = rgba[:, :, :3]     # RGB-каналы
                mask = rgba[:, :, 3]
                image = process_sprite(original, mask, nst, retinexnet, upsampler)

                name = base_name
                if i > 0:
                    name += '_' + str(i + 1)
                layer = PixelLayer.frompil(image, psd_main, name, 0, 0, Compression.RAW)
                group.append(layer)
        except NotAnImageException as e:
            continue

    add_max_size_layer(group, args.max_width, args.max_height, style_image)

    print('packing...')
    pack_psd(group)

    print('saving...')
    psd_main.save(args.output)

if __name__ == "__main__":
    main()