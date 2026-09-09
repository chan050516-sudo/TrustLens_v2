import json
import math
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from rapidocr_onnxruntime import RapidOCR
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

# 引入 PaddleOCR 的版面分析模块
from paddleocr import PPStructureV3


def deskew_image_and_ocr(image_cv, ocr_engine):
    """运行 OCR 粗扫，利用文本多边形计算中位数倾角并对图像与 OCR 框同步拉正"""
    h, w = image_cv.shape[:2]
    ocr_results, _ = ocr_engine(image_cv)

    angles = []
    if ocr_results:
        for poly, text, conf in ocr_results:
            if len(text.strip()) >= 4:
                # poly: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
                dx = poly[1][0] - poly[0][0]
                dy = poly[1][1] - poly[0][1]
                angle = math.degrees(math.atan2(dy, dx))
                if -15.0 < angle < 15.0:
                    angles.append(angle)

    median_angle = float(np.median(angles)) if angles else 0.0

    if abs(median_angle) < 0.2:
        print(f"   [Deskew] 倾斜度较小 ({median_angle:.2f}°)，无需回正。")
        return image_cv, ocr_results, 0.0

    print(f"   [Deskew] 检测到图片倾角: {median_angle:.2f}°，执行仿射拉正...")
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed_cv = cv2.warpAffine(
        image_cv,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # 在拉正后的图上重新获取高精度的水平 OCR 结果
    deskewed_ocr, _ = ocr_engine(deskewed_cv)
    return deskewed_cv, deskewed_ocr, median_angle


def main():
    image_path = (
        r"C:\Users\chan0\Personal"
        r" Project\TrustLens_v2\engine\test_doc\invoice_test1.jpg"
    )

    print("1. 载入图像并执行 OCR 倾角校正 (Deskew)...")
    ocr = RapidOCR()
    raw_cv = cv2.imread(image_path)
    if raw_cv is None:
        print(f"错误: 无法读取文件 {image_path}")
        return

    deskewed_cv, ocr_results, angle = deskew_image_and_ocr(raw_cv, ocr)

    # 转回 PIL 用于后续模型推理与绘制
    deskewed_rgb = cv2.cvtColor(deskewed_cv, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(deskewed_rgb)
    W, H = image.size

    print("2. 加载 PP-DocLayoutV3 模型并在拉正图上定位主表...")
    # 初始化 PP-Structure：关闭恢复排版、表格结构提取和 OCR，仅开启版面检测
    layout_engine = PPStructureV3(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )
    
    # PPStructure 接受 cv2 图像输入
    layout_results = layout_engine(deskewed_cv)

    table_boxes = []
    # 遍历版面解析结果，筛选出类别为 'table' 的物理框
    for region in layout_results:
        if region['type'].lower() == 'table':
            table_boxes.append([round(i, 2) for i in region['bbox']])

    if not table_boxes:
        print("PP-DocLayoutV3 未检测到表格区域，退出。")
        return

    # 选取面积最大的表格切片（排除可能的印章或其他小表格干扰）
    table_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    tx1, ty1, tx2, ty2 = table_boxes[0]
    print(f"   => PP-DocLayout 命中主表格物理坐标: [{tx1}, {ty1}, {tx2}, {ty2}]")

    # 裁剪表格区域（留出微小 padding）
    padding = 6
    crop_x1, crop_y1 = max(0, tx1 - padding), max(0, ty1 - padding)
    crop_x2, crop_y2 = min(W, tx2 + padding), min(H, ty2 + padding)
    cropped_table = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    crop_w, crop_h = cropped_table.size

    print("3. 加载 TATR Structure 模型解析行列网格...")
    struct_processor = AutoImageProcessor.from_pretrained(
        "microsoft/table-transformer-structure-recognition"
    )
    struct_model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-structure-recognition"
    )

    struct_inputs = struct_processor(images=cropped_table, return_tensors="pt")
    with torch.no_grad():
        struct_outputs = struct_model(**struct_inputs)

    struct_results = struct_processor.post_process_object_detection(
        struct_outputs,
        threshold=0.6,
        target_sizes=torch.tensor([[crop_h, crop_w]]),
    )[0]

    rows, cols = [], []
    id2label = struct_model.config.id2label

    for score, label, box in zip(
        struct_results["scores"], struct_results["labels"], struct_results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        # 将相对切片的坐标转换回全图物理绝对坐标
        global_box = [
            box[0] + crop_x1,
            box[1] + crop_y1,
            box[2] + crop_x1,
            box[3] + crop_y1,
        ]

        lbl = id2label[label.item()]
        if lbl == "table row":
            rows.append(global_box)
        elif lbl == "table column":
            cols.append(global_box)

    # 严格空间排序
    rows.sort(key=lambda x: x[1])
    cols.sort(key=lambda x: x[0])
    print(f"   => 解析网格结构：{len(rows)} 行, {len(cols)} 列")

    print("4. 将文字行质心投影到 Cell 单元格中...")
    table_data = []
    for r_idx, r_box in enumerate(rows):
        for c_idx, c_box in enumerate(cols):
            cell_xmin = max(r_box[0], c_box[0])
            cell_ymin = max(r_box[1], c_box[1])
            cell_xmax = min(r_box[2], c_box[2])
            cell_ymax = min(r_box[3], c_box[3])

            cell_texts = []
            if ocr_results:
                for poly, text, conf in ocr_results:
                    cx = sum([p[0] for p in poly]) / 4
                    cy = sum([p[1] for p in poly]) / 4
                    if (cell_xmin <= cx <= cell_xmax) and (cell_ymin <= cy <= cell_ymax):
                        cell_texts.append(text)

            table_data.append({
                "row_idx": r_idx,
                "col_idx": c_idx,
                "text": " ".join(cell_texts),
                "cell_bbox": [cell_xmin, cell_ymin, cell_xmax, cell_ymax],
            })

    with open("table_output_pplayout.json", "w", encoding="utf-8") as f:
        json.dump(table_data, f, indent=4, ensure_ascii=False)

    draw = ImageDraw.Draw(image)
    # 绿色主表边界 (PP-DocLayout 预测)
    draw.rectangle([tx1, ty1, tx2, ty2], outline="green", width=3)
    # 红色垂直列
    for c in cols:
        draw.rectangle(c, outline="red", width=2)
    # 蓝色水平行
    for r in rows:
        draw.rectangle(r, outline="blue", width=2)

    image.save("table_grid_pplayout_visualized.jpg")
    print(
        "✅ 运行完成！结果已保存至 table_grid_pplayout_visualized.jpg 和"
        " table_output_pplayout.json"
    )


if __name__ == "__main__":
    main()