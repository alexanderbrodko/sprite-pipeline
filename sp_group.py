import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from modelscope import AutoModelForImageSegmentation
import gdown
import os
import argparse
from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer, Compression
from PIL import Image
from sp_pack import pack_psd
from tqdm import tqdm
import configparser

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

def layer_matches_pattern(layer_name, pattern):
    """
    Проверяет, подходит ли имя слоя под паттерн
    Поддерживает * в конце как в примере
    """
    if pattern.endswith('*'):
        return layer_name.startswith(pattern[:-1])
    else:
        return layer_name == pattern

def get_layer_params(layer_name, config):
    """
    Возвращает масштаб ресайза и коэффициент relight для слоя из INI конфига
    """
    resize_scale = 1.0
    relight_coef = 0.3
    shadow_type = None
    gamma = 1.0
    
    if config and 'resize' in config:
        for pattern in config['resize']:
            if layer_matches_pattern(layer_name, pattern):
                resize_scale = float(config['resize'][pattern])
                break
    
    if config and 'relight' in config:
        for pattern in config['relight']:
            if layer_matches_pattern(layer_name, pattern):
                relight_coef = float(config['relight'][pattern])
                break

    if config and 'shadows' in config:
        for pattern in config['shadows']:
            if layer_matches_pattern(layer_name, pattern):
                shadow_type = config['shadows'][pattern].strip('"').lower()
                # Default to flip if value is empty or invalid
                if shadow_type and shadow_type not in ['flip', 'straight']:
                    print(f'wrong {shadow_type=}, must be flip or straight')
                    shadow_type = None
                break

    if config and 'gamma' in config:
        for pattern in config['gamma']:
            if layer_matches_pattern(layer_name, pattern):
                gamma = float(config['gamma'][pattern])
                break
    
    return resize_scale, relight_coef, shadow_type, gamma

def resize_rgba(rgba, scale):
    """
    Ресайзит RGBA массив с указанным масштабом
    """
    if scale is None or scale == 1.0:
        return rgba
    
    h, w = rgba.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Ресайзим с тем же интерполятором, что и в основном коде
    resized_rgba = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return resized_rgba

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
        output_array = (output_array * 255.0).astype(np.uint8)
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

def fit_size(w, h, max_w, max_h):
    scale_w = max_w / w
    scale_h = max_h / h
    scale = min(scale_w, scale_h, 1.0)
    return int(w * scale), int(h * scale)

def extract_sprites(rgba: np.ndarray, segmenter=None):
    mask = rgba[:, :, 3]
    if np.all(mask == 255):
        mask = segmenter(rgba[:,:,:3])
        mask = cv2.resize(mask, (rgba.shape[1], rgba.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    
    if mask is not None:
        rgba[:, :, 3] = mask
    
    if np.all(mask == 255):
        return [rgba]

    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    sprites = []

    for stat in stats[1:]:  # Пропускаем фон (статистика индекса 0)
        x, y, w, h, area = stat
        if area < 100:  # Фильтруем слишком маленькие области
            continue

        if h < 32 or w < 32:  # Если сторона меньше — пропускаем
            continue

        sprite_rgba = rgba[y:y+h, x:x+w]
        sprites.append(sprite_rgba)

    return sprites

def gen_shadow(rgba, shadow_type='flip'):
    """
    Generate shadow effect from RGBA image
    shadow_type: 'flip' or 'straight'
    """
    h, w = rgba.shape[:2]
    
    # 1) Resize to half height (for both types)
    new_h = max(1, h // 2)
    shadow_rgba = cv2.resize(rgba, (w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 2) Add padding
    pad = 20
    shadow_rgba = cv2.copyMakeBorder(shadow_rgba, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    
    # 3) Flip vertically only for 'flip' type
    if shadow_type == 'flip':
        shadow_rgba = cv2.flip(shadow_rgba, 0)
    
    # 4) Split into RGB and alpha for separate processing
    shadow_rgb = shadow_rgba[:, :, :3]
    shadow_alpha = shadow_rgba[:, :, 3]
    
    # 5) Gaussian blur with different sigmas for RGB and alpha
    shadow_rgb = cv2.GaussianBlur(shadow_rgb, (0, 0), sigmaX=16, sigmaY=16)
    shadow_alpha = cv2.GaussianBlur(shadow_alpha, (0, 0), sigmaX=8, sigmaY=8)
    
    # 6) Make alpha fuzzy
    shadow_alpha = shadow_alpha.astype(np.float32)
    shadow_alpha = np.maximum(0.0, np.power(shadow_alpha / 255.0, 0.75) * 255.0 - 70.0)
    shadow_alpha = shadow_alpha.astype(np.uint8)
    
    # 7) Brightness to almost dark
    shadow_rgb = shadow_rgb.astype(np.float32)
    shadow_rgb = np.power(shadow_rgb / 255.0, 0.75) * 255.0 * 0.25
    shadow_rgb = shadow_rgb.astype(np.uint8)
    
    # 8) Opacity 50%
    shadow_alpha = (shadow_alpha * 0.5).astype(np.uint8)
    
    # Combine back to RGBA
    shadow_rgba = np.dstack([shadow_rgb, shadow_alpha])

    h, w = shadow_rgba.shape[:2]
    
    return cv2.resize(shadow_rgba, (w // 2, h // 2), interpolation=cv2.INTER_LANCZOS4)

def new_layer(group, rgba, psd, name, retinexnet, debug, max_width, max_height,
        resize_scale=1.0, relight_coef=1.0, shadow_type=None, gamma=1.0):

    original = rgba[:, :, :3]
    mask = rgba[:, :, 3]
    
    # сначала ресайз, потом фильтры
    h, w = rgba.shape[:2]
    
    # Сначала применяем ограничение максимальных размеров
    new_w, new_h = fit_size(w, h, max_width, max_height)
    scale_fit = min(new_w / w, new_h / h)
    
    # Затем применяем правило из файла (если есть)
    final_scale = scale_fit
    final_scale = final_scale * resize_scale
    
    # Применяем финальный ресайз
    if final_scale != 1.0:
        if debug and final_scale > 1.0:
            print(f"WARNING: {name} upscaled {final_scale:.2f}x (original: {w}x{h})")

        final_w = int(w * final_scale)
        final_h = int(h * final_scale)
        # Ресайзим и оригинал и маску
        original = cv2.resize(original, (final_w, final_h), interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.resize(mask, (final_w, final_h), interpolation=cv2.INTER_LANCZOS4)
        if debug:
            print(f"Resized {name} with scale {final_scale:.2f} ({scale_fit=:.2f}, {resize_scale=}, {relight_coef=})")
    
    # фильтры применяем к уже ресайзнутому изображению
    result_retinexnet = flat_lights(original, retinexnet).astype(np.float32)
    original = original.astype(np.float32)
    
    # Применяем коэффициент relight к смешиванию
    corrected = (original * (1 - relight_coef) + result_retinexnet * relight_coef).astype(np.uint8)

    if gamma != 1.0:
        raw = corrected.astype(np.float32) / 255.0
        corrected = np.clip(np.power(raw, gamma) * 255.0, 0, 255.0).astype(np.uint8)
    
    rgba = np.dstack([corrected, mask])

    image = Image.fromarray(rgba, mode="RGBA")
    
    if debug:
        image.save(os.path.join('debug', name + '.png'))
    
    layer = PixelLayer.frompil(image, psd, name, 0, 0, Compression.RAW)
    group.append(layer)

    if shadow_type:
        shadow_image_np = gen_shadow(rgba, shadow_type)
        shadow_image_pil = Image.fromarray(shadow_image_np, mode="RGBA")

        layer = PixelLayer.frompil(shadow_image_pil, psd, name + '_shadow', 0, 0, Compression.RAW)
        group.append(layer)

def main():
    model_dir = os.path.expanduser('~/.cache/sprite-pipeline/')
    decom_path = os.path.join(model_dir, 'models/decom.tar')
    relight_path = os.path.join(model_dir, 'models/relight.tar')
    if not (
        os.path.isfile(decom_path) or
        os.path.isfile(relight_path)
    ):
        drive_path = 'https://drive.google.com/drive/folders/1gxAukn_M7YNbnWfg_OrV6BxlNWUuDPmL'
        gdown.download_folder(url=drive_path, output=model_dir, quiet=False, use_cookies=False)

    parser = argparse.ArgumentParser(description="Make PSD group from folder with sprites.")
    parser.add_argument("folder", help="Path to folder with images.")
    parser.add_argument("-W", "--max_width", type=int, default=480, help="Max sprite width.")
    parser.add_argument("-H", "--max_height", type=int, default=480, help="Max sprite height.")
    parser.add_argument("-o", "--output", default="output.psd", help="Output PSD name.")
    parser.add_argument("--config", help="Path to config INI. Sections: resize, relight")
    parser.add_argument("--debug", action="store_true", help="Output each sprite in ./debug/.")
    args = parser.parse_args()

    # Загружаем правила ресайза если файл указан
    config = []
    if args.config and os.path.exists(args.config):
        config = configparser.ConfigParser()
        config.read(args.config)

    retinexnet = RetinexNetWrapper(decom_path, relight_path).to(device)

    if device == torch.device('cpu'):
        print('Warning: torch device is CPU so foreground extraction is ultra slow.')

    birefnet = AutoModelForImageSegmentation.from_pretrained('modelscope/BiRefNet', trust_remote_code=True)
    birefnet.to(device).eval().half()
    segmenter = lambda image_np: extract_foreground_mask_birefnet(image_np, birefnet)

    try:
        psd_main = PSDImage.open(args.output)
    except Exception as e:
        psd_main = PSDImage.new(mode='RGBA', size=(1000, 1000))

    if args.debug:
        if os.path.exists('debug'):
            for filename in os.listdir('debug'):
                os.remove(os.path.join('debug', filename))
        os.makedirs('debug', exist_ok=True)
    
    progress_bar = tqdm(total=len(os.listdir(args.folder)), desc='extract')

    group_name = os.path.basename(os.path.normpath(args.folder))
    
    for layer in psd_main:
        if layer.name == group_name:
            psd_main.remove(layer)
            break
    
    group = Group.new(group_name, open_folder=False, parent=psd_main)
    for filename in os.listdir(args.folder):
        progress_bar.update()
        image_path = os.path.join(args.folder, filename)

        rgba = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if rgba is None:
            print('\rSkip', filename, ' - can not load')
            continue

        rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
            
        sprites = extract_sprites(rgba, segmenter)

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        for i, sprite_rgba in enumerate(sprites):
            name = base_name
            if i > 0:
                name += '_' + str(i + 1)
            
            resize_scale, relight_coef, shadow_type, gamma = get_layer_params(name, config)

            # Передаем max_width, max_height и relight_coef в new_layer
            new_layer(group, sprite_rgba, psd_main, name, retinexnet, args.debug, 
                      args.max_width, args.max_height, resize_scale, relight_coef, shadow_type, gamma)

            
    progress_bar.close()

    print('packing...')
    pack_psd(group)

    print('saving...')
    psd_main.save(args.output)

if __name__ == "__main__":
    main()
    