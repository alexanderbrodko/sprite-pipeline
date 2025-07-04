from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer, Compression
from psd_tools.constants import BlendMode
from rectpack import newPacker
from PIL import Image, ImageOps
import argparse
import numpy as np
import cv2
import os
import json

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
        bordered_image = ImageOps.expand(cropped_image, border=10, fill=(255,255,255, 0))
        
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
        if canvas_size[1] < canvas_size[0]:
            canvas_size[1] *= 2
        else:
            canvas_size[0] *= 2
        print('repack to', canvas_size)
        packer, all_fitted = pack_layers(layers_info, canvas_size)
    
    repack_layers(psd, packer, layers_info)

def create_uv_json(psd, width, height, output_path):
    data = {}
    
    for layer in psd:
        if layer.is_group():
            continue
        
        data[layer.name] = {}

        data[layer.name]['left'] = layer.left
        data[layer.name]['top'] = layer.top
        data[layer.name]['right'] = layer.right
        data[layer.name]['bottom'] = layer.bottom
    
    # Сохраняем в JSON-файл
    with open(output_path, 'w') as f:
        json.dump(data, f)

def makeShadow(
    image_np,
    radius=50,
    smooth=3,
    scale=0.1,
    color=(0.04, 0.03, 0.06),
    noise_level=0.05,
    intensity=0.3
):
    if image_np.shape[2] != 4:
        raise ValueError("Изображение должно быть в формате RGBA")

    h, w = image_np.shape[:2]

    pad = radius * 2
    image_padded = cv2.copyMakeBorder(
        image_np,
        pad, pad, pad, pad,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0, 0)
    )

    # Извлекаем альфа-канал
    alpha_channel = image_padded[:, :, 3].astype(np.float32) / 255.0

    # Размываем альфа-канал
    blurred_alpha = cv2.GaussianBlur(alpha_channel, (radius * 2 + 1, radius * 2 + 1), radius)

    # Применяем степенную функцию для изменения контраста
    transformed_alpha = np.clip(blurred_alpha * smooth, 0, 1)
    transformed_alpha = cv2.GaussianBlur(transformed_alpha, (radius * 2 + 1, radius * 2 + 1), radius)

    # Масштабируем интенсивность тени
    transformed_alpha *= intensity

    shadow_alpha = (transformed_alpha * 255).astype(np.uint8)

    # Создаём RGB-изображение с указанным цветом тени
    shadow_rgb = np.zeros((image_padded.shape[0], image_padded.shape[1], 3), dtype=np.uint8)
    shadow_rgb[:, :, :] = color  # Задаём цвет тени

    # Добавляем шум (если указано)
    if noise_level > 0.0:
        noise = np.random.normal(0, noise_level, shadow_rgb.shape).astype(np.float32)
        shadow_rgb = np.clip(shadow_rgb.astype(np.float32) / 255 + noise, 0, 1) * 255
        shadow_rgb = shadow_rgb.astype(np.uint8)

    # Накладываем альфа-канал на RGB
    shadow_rgb = cv2.merge([shadow_rgb, shadow_alpha])

    # Сжимаем изображение
    new_size = (int(shadow_rgb.shape[1] * scale), int(shadow_rgb.shape[0] * scale))
    shadow_scaled = cv2.resize(shadow_rgb, new_size, interpolation=cv2.INTER_AREA)

    return shadow_scaled

def add_shadow_layers(group, psd):
    root = psd  # Корень PSD

    shadow_layers = []

    for layer in group:
        if not layer.visible:  # Пропускаем невидимые слои
            continue

        img_np = np.array(layer.composite(force=True))

        # Генерируем тень
        shadow_np = makeShadow(img_np, radius=5, smooth=8, scale=0.5, intensity=0.66)

        shadow_pil = Image.fromarray(shadow_np)

        # Создаём новый слой тени и добавляем его в корень PSD
        shadow_layer = PixelLayer.frompil(
            shadow_pil,
            root,
            f"{layer.name}_shadow",
            0, 0,
            Compression.RAW
        )
        shadow_layers.append(shadow_layer)

    for layer in shadow_layers:
        group.append(layer)


def main():
    parser = argparse.ArgumentParser(description="Make spritesheets from PSD.")
    parser.add_argument("psd", nargs='?', default="output.psd", help="Path to your PSD with groups.")
    parser.add_argument("-o", "--output_dir", default='.', help="Output dir.")
    parser.add_argument("--format", default='png', help="Output image format.")
    parser.add_argument("--gen_shadows", action="store_true", help="Generate shadows for sprites.")

    args = parser.parse_args()

    psd_main = PSDImage.open(args.psd)

    for group in psd_main:
        if not group.is_group():
            continue

        if args.gen_shadows:
            print('gen shadows for', group.name + '... ', end="")
            add_shadow_layers(group, psd_main)
            print('done')

        layers_info = process_layers(group)

        pack_psd(group)

        image = group.composite(force=True)
        bbox = image.getbbox()
        if bbox is None:
            continue
        cropped_image = image.crop(bbox)
        image = ImageOps.expand(cropped_image, border=10, fill=(255,255,255, 0))
        
        png_path = os.path.join(args.output_dir, group.name + '.' + args.format)
        image.save(png_path)
        txt_path = os.path.join(args.output_dir, group.name + '_uv.json')
        create_uv_json(group, image.width, image.height, txt_path)

if __name__ == "__main__":
    main()