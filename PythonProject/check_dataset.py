"""检查 VOC2007 训练集完整性"""
import os
import xml.etree.ElementTree as ET
import cv2

ROOT = 'data/VOCdevkit/VOC2007'
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

for split in ['train', 'val']:
    print(f"\n{'='*60}")
    print(f"检查 {split} 集")
    print(f"{'='*60}")

    with open(os.path.join(ROOT, f'ImageSets/Main/{split}.txt')) as f:
        ids = [line.strip() for line in f.readlines() if line.strip()]
    print(f"{split}.txt 中的图片数: {len(ids)}")
    print(f"唯一 ID 数: {len(set(ids))}（如有重复说明文件有问题）")

    missing_img, missing_xml = [], []
    empty_anno, bad_box, unreadable_img = [], [], []
    class_count = {c: 0 for c in CLASSES}
    total_objects = 0
    box_sizes = []  # (w, h) 归一化

    for img_id in ids:
        img_path = os.path.join(ROOT, f'JPEGImages/{img_id}.jpg')
        xml_path = os.path.join(ROOT, f'Annotations/{img_id}.xml')

        if not os.path.exists(img_path):
            missing_img.append(img_id)
            continue
        if not os.path.exists(xml_path):
            missing_xml.append(img_id)
            continue

        # 检查图片可读
        img = cv2.imread(img_path)
        if img is None:
            unreadable_img.append(img_id)
            continue
        h, w = img.shape[:2]

        # 检查标注
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            bad_box.append((img_id, 'XML解析失败'))
            continue
        root = tree.getroot()
        objs = root.findall('object')
        if len(objs) == 0:
            empty_anno.append(img_id)
            continue

        for obj in objs:
            name = obj.find('name').text
            if name not in CLASSES:
                bad_box.append((img_id, f'未知类别: {name}'))
                continue
            class_count[name] += 1
            total_objects += 1
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            if xmax <= xmin or ymax <= ymin:
                bad_box.append((img_id, f'非法框: ({xmin},{ymin},{xmax},{ymax})'))
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            box_sizes.append((bw, bh))

    print(f"\n缺少图片文件: {len(missing_img)}")
    print(f"缺少标注文件: {len(missing_xml)}")
    print(f"图片无法读取: {len(unreadable_img)}")
    print(f"空标注(无object): {len(empty_anno)}")
    print(f"非法标注: {len(bad_box)}")
    if bad_box[:5]:
        for b in bad_box[:5]:
            print(f"  {b}")

    print(f"\n总目标数: {total_objects}")
    print(f"平均每张图目标数: {total_objects/max(len(ids),1):.2f}")

    # 框尺寸分布
    if box_sizes:
        import numpy as np
        ws = np.array([b[0] for b in box_sizes])
        hs = np.array([b[1] for b in box_sizes])
        print(f"\nGT 框宽度(归一化): min={ws.min():.3f}, max={ws.max():.3f}, mean={ws.mean():.3f}")
        print(f"GT 框高度(归一化): min={hs.min():.3f}, max={hs.max():.3f}, mean={hs.mean():.3f}")
        print(f"GT 框宽高比(w/h): min={(ws/hs).min():.2f}, max={(ws/hs).max():.2f}")
        # 尺寸分桶
        areas = ws * hs * 300 * 300  # 像素面积
        print(f"\n目标像素面积分布 (300x300尺度):")
        print(f"  <32x32 (小): {(areas < 32*32).sum()}")
        print(f"  32x32~96x96 (中): {((areas >= 32*32) & (areas < 96*96)).sum()}")
        print(f"  >96x96 (大): {(areas >= 96*96).sum()}")

    print(f"\n各类别数量:")
    for c in CLASSES:
        print(f"  {c:12s}: {class_count[c]}")
