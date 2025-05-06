import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
import os
import argparse
from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer, Compression
from psd_tools.constants import BlendMode
from PIL import Image, ImageOps
from nst_vgg19 import NST_VGG19
from rectpack import newPacker
from retinex import msrcr
from modelscope import AutoModelForImageSegmentation
from modelscope import snapshot_download
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

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
    # Проверка существования файла
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Файл не найден: {img_path}")

    # Чтение изображения
    original = cv2.imread(img_path)
    if original is None:
        raise ValueError(f"Файл не является изображением или поврежден: {img_path}")

    print(img_path)
    
    # Удаление альфа-канала, если он есть
    if len(original.shape) == 3 and original.shape[2] == 4:  # Проверка наличия альфа-канала
        original = original[:, :, :3]
        print(f'^skip alpha')

    # Преобразование цветового пространства BGR -> RGB
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Уменьшение размера, если изображение слишком большое
    original_height, original_width = original.shape[:2]
    new_width, new_height = get_max_size(original_width, original_height, max_width, max_height)
    if original_width != new_width or original_height != new_height:
        original = cv2.resize(original, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    return original

def apply_mask(image_np, mask_np):
    """
    Применяет маску к изображению.
    
    Args:
        image_np (np.ndarray): Исходное изображение в виде Numpy-массива (H, W, C) в диапазоне [0, 255].
        mask_np (np.ndarray): Маска в виде Numpy-массива (H, W) или (H, W, 1) в диапазоне [0, 255].
    
    Returns:
        np.ndarray: Изображение с примененной маской в виде Numpy-массива (H, W, 4) в формате RGBA.
    """
    # Убедимся, что маска имеет тот же размер, что и изображение
    if mask_np.shape[:2] != image_np.shape[:2]:
        raise ValueError("Размеры изображения и маски должны совпадать.")
    
    # Если маска одноканальная (grayscale), добавляем канал
    if len(mask_np.shape) == 2:
        mask_np = np.expand_dims(mask_np, axis=2)  # (H, W) -> (H, W, 1)
    
    # Добавляем альфа-канал к изображению
    if image_np.shape[2] == 3:  # Если изображение RGB
        image_with_alpha = np.concatenate(
            [image_np, mask_np], axis=2
        ).astype(np.uint8)
    else:
        raise ValueError("Изображение должно быть в формате RGB (H, W, 3).")
    
    return image_with_alpha

def clahe(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l_channel = clahe.apply(l_channel)
    enhanced_lab_image = cv2.merge((enhanced_l_channel, a_channel, b_channel))
    enhanced_rgb_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2RGB)
    return enhanced_rgb_image

def add_max_size_layer(psd, max_width, max_height, color=(255, 0, 0)):
    for layer in psd:
        if layer.name == 'max_size':
            return

    image = Image.fromarray(np.full((max_height, max_width, 3), color, dtype=np.uint8), mode='RGB')

    root = psd
    while root.parent:
        root = root.parent

    new_layer = PixelLayer.frompil(image, root, 'max_size', 0, 0, Compression.RAW)

    psd.append(new_layer)

def process_image(img_path, max_width, max_height, nst, psd, edgePreservingFilter_sigma_s=0, retinexnet=None, birefnet=None, refiner=None):
    try:
        original = load_image(img_path, max_width * 2, max_height * 2)
    except Exception as e:
        return
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = 'sprites'
    output_path = os.path.join(output_dir, f"{base_name}")
    os.makedirs(output_dir, exist_ok=True)

    original = clahe(original)

    corrected = original
    #height, width = corrected.shape[:2]
    #corrected = cv2.resize(corrected, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
    
    corrected = flat_lights(original, retinexnet)

    enhanced = msrcr(original, sigmas=[15, 80, 250])

    corrected = (corrected.astype(np.float32) * 0.6 + enhanced.astype(np.float32) * 0.4).astype(np.uint8)

    if edgePreservingFilter_sigma_s >= 1:
        corrected = cv2.edgePreservingFilter(corrected, flags=2, sigma_s=edgePreservingFilter_sigma_s, sigma_r=0.4)

    if nst is not None:
        corrected = nst(corrected)
        
    height, width = corrected.shape[:2]
    corrected, _ = refiner.enhance(corrected)
    corrected = cv2.resize(corrected, (width, height), interpolation=cv2.INTER_LANCZOS4)

    mask = extract_foreground_mask(original, birefnet)

    corrected_rgba = apply_mask(corrected, mask)

    image = Image.fromarray(corrected_rgba, mode="RGBA")

    image.save(output_path + '.png')
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

def process_layers(psd):
    layers_info = []
    for layer in psd:
        if layer.is_group():
            continue  # Пропускаем группы слоев
        if not layer.visible:
            continue  # Пропускаем невидимые слои

        # Преобразуем слой в изображение PIL
        image = layer.composite(force=True)

        # Шаг 1: Обрезаем слой до содержимого
        bbox = image.getbbox()
        if bbox is None:
            continue  # Пропускаем пустые слои
        cropped_image = image.crop(bbox)
        
        # Шаг 2: Добавляем рамку в 1 прозрачный пиксель
        bordered_image = ImageOps.expand(cropped_image, border=1, fill=(0, 0, 0, 0))
        bordered_image.save(f'sprites/{layer.name}.png')
        
        # Сохраняем информацию о слое
        layers_info.append({
            "layer": layer,
            "name": layer.name,
            "image": bordered_image,
            "size": bordered_image.size,
        })

    return layers_info

def pack_layers(layers, canvas_size):
    packer = newPacker(rotation=False)  # Запрещаем поворот
    
    # Добавляем прямоугольники с уникальными идентификаторами
    for i, info in enumerate(layers):
        packer.add_rect(*info["size"], rid=i)  # rid - уникальный идентификатор прямоугольника
    
    packer.add_bin(*canvas_size)
    packer.add_bin(*canvas_size)
    packer.pack()
    
    # Проверяем, все ли слои влезли
    all_fitted = True
    for rect in packer.rect_list():
        b, x, y, w, h, rid = rect
        if w == 0 or h == 0 or b != 0:
            all_fitted = False
            break
    
    return packer, all_fitted

def repack_layers(psd, packer, layers_info):
    # Создаем новые слои на основе упаковки
    for rect in packer.rect_list():
        b, x, y, w, h, rid = rect
        layer_info = layers_info[rid]  # Используем идентификатор b для получения слоя
        old_layer = layer_info["layer"]

        root = psd
        while root.parent:
            root = root.parent

        # Создаем новый слой из PIL-изображения
        new_layer = PixelLayer.frompil(
            layer_info["image"],
            root,
            layer_info["name"],
            y,
            x,
            Compression.RAW
        )
        # Устанавливаем атрибуты нового слоя
        new_layer.visible = True  # Убедитесь, что слой видим
        new_layer.opacity = 255  # Полная непрозрачность
        new_layer.blend_mode = BlendMode.NORMAL  # Режим наложения "normal"

        # Находим индекс старого слоя
        try:
            index = psd.index(old_layer)
        except ValueError:
            # Если слой не найден, добавляем новый слой в конец
            print(f"Layer '{layer_info['name']}' not found in PSD, appending to the end.")
            psd.append(new_layer)
            continue
        
        # Удаляем старый слой
        psd.remove(old_layer)
        
        # Вставляем новый слой на то же место
        psd.insert(index, new_layer)

def pack_psd(psd):
    # Инициализация размера холста (используем список для изменения)
    canvas_size = [512, 512]
    
    # Список для хранения информации о слоях
    layers_info = process_layers(psd)
    
    # Упаковываем слои
    packer, all_fitted = pack_layers(layers_info, canvas_size)
    
    # Если слои не влезли, увеличиваем минимальную сторону холста в 2 раза
    while not all_fitted:
        print('repack')
        if canvas_size[1] < canvas_size[0]:
            canvas_size[1] *= 2
        else:
            canvas_size[0] *= 2
        packer, all_fitted = pack_layers(layers_info, canvas_size)
    
    repack_layers(psd, packer, layers_info)

def create_uv_file(psd, name):
    """
    Создает файл uv.txt с именами всех слоев и их UV-координатами.
    
    :param psd: Исходный PSD-файл.
    """
    layer_names = []  # Список для хранения имен слоев
    uv_coordinates = []  # Список для хранения UV-координат
    
    for layer in psd:
        if layer.is_group():
            continue  # Пропускаем группы слоев
        
        # Сохраняем имя слоя
        layer_names.append(layer.name)
        
        # Вычисляем UV-координаты
        u0 = layer.left / psd.width
        v0 = layer.top / psd.height
        u1 = layer.right / psd.width
        v1 = layer.bottom / psd.height
        
        # Добавляем UV-координаты в список
        uv_coordinates.extend([
            u0, v0,  # Левый верхний угол
            u1, v0,  # Правый верхний угол
            u1, v1,  # Правый нижний угол
            u0, v1   # Левый нижний угол
        ])
    
    # Создаем файл uv.txt
    with open(name + "_uv.txt", "w") as uv_file:
        # Первая строка: имена слоев через запятую
        uv_file.write(",".join(layer_names) + "\n")
        
        # Вторая строка: UV-координаты через запятую
        uv_file.write(",".join(map(str, uv_coordinates)) + "\n")

if __name__ == "__main__":
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Apply RetinexNet, SAM, and style transfer to images.")
    parser.add_argument("--style", required=False, help="Path to the style image.")
    parser.add_argument("folder", help="Path to folder with images.")
    parser.add_argument("-W", "--max_width", type=int, default=512, help="Max sprite width.")
    parser.add_argument("-H", "--max_height", type=int, default=512, help="Max sprite height.")
    parser.add_argument("-f", "--nst_force", type=float, default=1, help="Neural style transfer - weights mul.")
    parser.add_argument("-b", "--edgepreservingfilter_sigma_s", type=float, default=0, help="Blur 1 - 200.0. For edgePreservingFilter.")
    parser.add_argument("-o", "--output", default="output.psd", help="Output PSD name.")
    args = parser.parse_args()

    # Загрузка модели стиля, если указан стиль
    nst = None
    if args.style:
        style_image = load_image(args.style)
        mul = args.nst_force
        STYLE_WEIGHTS = {
            'conv_2': 0.000001 * mul,  # Light/shadow?
            'conv_4': 0.000009 * mul,  # Contrast?
            'conv_5': 0.000006 * mul,  # Volume?
            'conv_7': 0.000003 * mul,
            'conv_8': 0.000002 * mul,  # Dents?
            'conv_9': 0.000003 * mul,
            'conv_11': 0.000001 * mul,
            'conv_13': 0.000001 * mul,
            'conv_15': 0.000001 * mul,
        }
        nst = NST_VGG19(style_image, style_layers_weights=STYLE_WEIGHTS)

    try:
        psd_main = PSDImage.open(args.output)
    except Exception as e:
        psd_main = PSDImage.new(mode='RGBA', size=(1000, 1000))

    retinexnet = RetinexNetWrapper('decom.tar', 'relight.tar').to(device)

    birefnet = AutoModelForImageSegmentation.from_pretrained('modelscope/BiRefNet', trust_remote_code=True)
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    birefnet.to(device)
    birefnet.eval()
    birefnet.half()

    esrgan4plus = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)

    upsampler = RealESRGANer(
        scale=1,
        model_path='RealESRGAN_x2plus_mtg_v1.pth',
        dni_weight=None,
        model=esrgan4plus,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=None)

    group = Group.new(args.folder, open_folder=False, parent=psd_main)
    for filename in os.listdir(args.folder):
        image_path = os.path.join(args.folder, filename)
        process_image(image_path, args.max_width, args.max_height, nst, group, args.edgepreservingfilter_sigma_s, retinexnet, birefnet, upsampler)

    add_max_size_layer(group, args.max_width, args.max_height)

    print('packing...')
    pack_psd(group)

    psd_main.save(args.output)
