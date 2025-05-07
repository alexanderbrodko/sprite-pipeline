from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer, Compression
from psd_tools.constants import BlendMode
from rectpack import newPacker
from PIL import Image, ImageOps
import argparse
import os

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

def create_uv_file(psd, width, height, path):
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
        u0 = layer.left / width
        v0 = layer.top / height
        u1 = layer.right / width
        v1 = layer.bottom / height
        
        # Добавляем UV-координаты в список
        uv_coordinates.extend([
            u0, v0,  # Левый верхний угол
            u1, v0,  # Правый верхний угол
            u1, v1,  # Правый нижний угол
            u0, v1   # Левый нижний угол
        ])
    
    # Создаем файл uv.txt
    with open(path, "w") as uv_file:
        # Первая строка: имена слоев через запятую
        uv_file.write(",".join(layer_names) + "\n")
        
        # Вторая строка: UV-координаты через запятую
        uv_file.write(",".join(map(str, uv_coordinates)) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Make spritesheets from PSD.")
    parser.add_argument("psd", help="Path to your PSD with groups.")
    parser.add_argument("-o", "--output_dir", default='.', help="Output dir.")
    parser.add_argument("--format", default='png', help="Output image format.")
    args = parser.parse_args()

    psd_main = PSDImage.open(args.psd)

    for group in psd_main:
        if not group.is_group():
            continue
        pack_psd(group)

        image = group.composite(force=True)
        bbox = image.getbbox()
        if bbox is None:
            continue
        cropped_image = image.crop(bbox)
        image = ImageOps.expand(cropped_image, border=1, fill=(0, 0, 0, 0))

        png_path = os.path.join(args.output_dir, group.name + '.' + args.format)
        image.save(png_path)
        txt_path = os.path.join(args.output_dir, group.name + '_uv.txt')
        create_uv_file(group, image.width, image.height, txt_path)

if __name__ == "__main__":
    main()