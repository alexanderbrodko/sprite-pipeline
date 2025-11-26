from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer, Compression
from psd_tools.constants import BlendMode
from rectpack import newPacker, MaxRectsBaf, SORT_AREA
from PIL import Image, ImageOps
import argparse
import os
import json  # Добавляем импорт json

def get_layers_info(psd, border=10):
    layers_info = []
    for layer in psd:
        if layer.is_group():
            continue
        if not layer.visible:
            continue

        image = layer.composite(force=True)
        bbox = image.getbbox()
        if bbox is None:
            continue
            
        # Сохраняем смещение от обрезки
        crop_left, crop_top, crop_right, crop_bottom = bbox
        original_width, original_height = image.size
        
        cropped_image = image.crop(bbox)
        bordered_image = ImageOps.expand(cropped_image, border=border, fill=(255,255,255, 0))
        
        layers_info.append({
            "layer": layer,
            "name": layer.name,
            "image": bordered_image,
            "size": bordered_image.size,
            "crop_offset": (crop_left, crop_top),  # Смещение обрезки
            "original_size": (original_width, original_height),
            "border": border
        })

    return layers_info

def pack_layers(layers, canvas_size):
    packer = newPacker(
        rotation=False,
        pack_algo=MaxRectsBaf,
        sort_algo=SORT_AREA,
    )
    
    for i, info in enumerate(layers):
        packer.add_rect(*info["size"], rid=i)
    
    packer.add_bin(*canvas_size)
    packer.pack()
    
    all_fitted = len(packer.rect_list()) == len(layers)
    
    return packer, all_fitted


def repack_layers(psd, packer, layers_info):
    packed_layers_info = []  # Сохраняем информацию об упакованных слоях
    
    for rect in packer.rect_list():
        b, x, y, w, h, rid = rect
        layer_info = layers_info[rid]
        old_layer = layer_info["layer"]

        # Сохраняем информацию для UV
        layer_info["atlas_x"] = x
        layer_info["atlas_y"] = y
        packed_layers_info.append(layer_info)

        root = psd
        while root.parent:
            root = root.parent

        new_layer = PixelLayer.frompil(
            layer_info["image"],
            root,
            layer_info["name"],
            y,  # top
            x,  # left
            Compression.RAW
        )
        
        new_layer.visible = True
        new_layer.opacity = 255
        new_layer.blend_mode = BlendMode.NORMAL

        try:
            index = psd.index(old_layer)
            psd.remove(old_layer)
            psd.insert(index, new_layer)
        except ValueError:
            print(f"Layer '{layer_info['name']}' not found in PSD, appending to the end.")
            psd.append(new_layer)
    
    return packed_layers_info


def pack_psd(psd, border=50):
    canvas_size = [512, 512]
    layers_info = get_layers_info(psd, border)
    
    packer, all_fitted = pack_layers(layers_info, canvas_size)
    
    while not all_fitted:
        print(f'repack to {canvas_size[0]}x{canvas_size[1]}')
        if canvas_size[0] < canvas_size[1]:
            canvas_size[0] += 256
        else:
            canvas_size[1] += 256
        packer, all_fitted = pack_layers(layers_info, canvas_size)
    
    packed_layers_info = repack_layers(psd, packer, layers_info)
    return packed_layers_info, canvas_size

def create_uv_file(psd, width, height, path, image_border=0):
    """
    Создает файл uv.json с именами всех слоев и их UV-координатами.
    
    :param psd: Исходный PSD-файл.
    :param width: Ширина итогового изображения.
    :param height: Высота итогового изображения.
    :param path: Путь для сохранения файла.
    """
    uv_data = {}  # Словарь для хранения UV-данных
    
    for layer in psd:
        if layer.is_group():
            continue  # Пропускаем группы слоев
        
        # Сохраняем UV-координаты для каждого слоя
        uv_data[layer.name] = {
            "left": layer.left + image_border,
            "top": layer.top + image_border,
            "right": layer.right + image_border,
            "bottom": layer.bottom + image_border
        }
    
    # Создаем JSON файл
    with open(path, "w") as uv_file:
        json.dump(uv_data, uv_file, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Make spritesheets from PSD.")
    parser.add_argument("psd", nargs='?', default="output.psd", help="Path to your PSD with groups.")
    parser.add_argument("-o", "--output_dir", default='.', help="Output dir.")
    parser.add_argument("--format", default='png', help="Output image format.")
    args = parser.parse_args()

    psd_main = PSDImage.open(args.psd)
    
    for group in psd_main:
        if not group.is_group():
            continue
        packed_layers_info, canvas_size = pack_psd(group, border=1)

        image = group.composite(force=True)
        bbox = image.getbbox()
        if bbox is None:
            continue
        cropped_image = image.crop(bbox)
        image = ImageOps.expand(cropped_image, border=1, fill=(255,255,255, 0))
        
        png_path = os.path.join(args.output_dir, group.name + '.' + args.format)
        image.save(png_path)
        json_path = os.path.join(args.output_dir, group.name + '_uv.json')  # Изменяем расширение на .json
        create_uv_file(group, image.width, image.height, json_path, image_border=1)

if __name__ == "__main__":
    main()