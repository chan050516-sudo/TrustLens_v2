import sys
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from enum import IntEnum
from PIL import Image, ImageDraw
import easyocr
from scipy.signal import find_peaks
from sklearn.linear_model import RANSACRegressor

# =========================================================================
# 1. 核心数据结构与排版分级
# =========================================================================
class TypographyClass(IntEnum):
    RELIABLE = 1    # 大写字母、数字 (适合做基线锚点)
    UNCERTAIN = 2   # 降笔画字母 (g, j, p, q, y)、特殊排版符号
    EXCLUDED = 3    # 标点符号 (.,:;'") - 绝对不能参与基线计算

def get_typography_class(char: str) -> TypographyClass:
    if char in ".,:;'\"()[]{}-_+=*&^%$#@!\\/|<>~`":
        return TypographyClass.EXCLUDED
    if char in "gjpqyJQ":
        return TypographyClass.UNCERTAIN
    return TypographyClass.RELIABLE

@dataclass
class CharMeasurement:
    char_text: str
    typography_cls: TypographyClass
    ink_bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    ink_bottom_y: float                  # 真实的墨迹最低点
    segmentation_confidence: float       # 0.0 ~ 1.0
    
    def get_center_x(self):
        return self.ink_bbox[0] + self.ink_bbox[2] / 2.0

# =========================================================================
# 2. 图像形态学与物理测量算法
# =========================================================================
def robust_ink_segmentation(gray_line: np.ndarray) -> np.ndarray:
    """高保真墨迹分割：保留抗锯齿边缘"""
    p2, p98 = np.percentile(gray_line, (2, 98))
    img_rescale = np.clip((gray_line - p2) / (p98 - p2 + 1e-5) * 255, 0, 255).astype(np.uint8)
    bg = cv2.GaussianBlur(img_rescale, (31, 31), 0)
    diff = cv2.subtract(bg, img_rescale)
    _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return binary

def advanced_projection_segmentation(ink_mask: np.ndarray, expected_char_count: int) -> Tuple[List[Tuple[int, int]], float]:
    """基于一维高斯平滑与波谷探测的字符切分"""
    h_img, w_img = ink_mask.shape
    v_proj = np.sum(ink_mask, axis=0) / 255.0
    
    # 高斯平滑投影曲线，消除单像素噪点引起的伪切分
    smoothed_proj = cv2.GaussianBlur(v_proj.reshape(1, -1), (5, 1), 0).flatten()
    
    # 寻找波谷 (通过对负信号寻峰)
    valleys, _ = find_peaks(-smoothed_proj, distance=3, prominence=0.5)
    
    boundaries = [0] + list(valleys) + [w_img]
    segments = []
    for i in range(len(boundaries) - 1):
        sx, ex = boundaries[i], boundaries[i+1]
        if ex - sx > 2:  # 忽略极窄的碎片
            segments.append((sx, ex))
            
    # 计算切割置信度 (Segmentation Confidence)
    observed_count = len(segments)
    if expected_char_count <= 0:
        confidence = 0.0
    elif observed_count == expected_char_count:
        confidence = 0.95
    else:
        diff_ratio = abs(observed_count - expected_char_count) / expected_char_count
        confidence = max(0.0, 0.8 - diff_ratio)
        
    return segments, confidence

def get_dynamic_density_threshold(char_text: str, w_local: int) -> float:
    """对窄字符实施密度保护，防止 Y 轴过度收敛导致高度坍塌"""
    if char_text in set("1Il|i:;.,'"):
        return 1.0  
    return min(max(2.0, w_local * 0.10), 5.0) 

def extract_line_geometry(line_rgb: np.ndarray, ocr_text: str) -> List[CharMeasurement]:
    """基于拓扑对齐与弹性切分的严丝合缝物理提取"""
    gray = cv2.cvtColor(line_rgb, cv2.COLOR_RGB2GRAY)
    ink_mask = robust_ink_segmentation(gray)
    
    chars_to_match = [c for c in ocr_text if not c.isspace()]
    if not chars_to_match:
        return []

    # 1. 连通域提取
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w >= 1 and h >= 2:  
            boxes.append([x, y, x + w, y + h])
            
    if not boxes:
        return []

    # 2. X轴深度重叠合并 (严格防误杀: 仅合并真正的上下堆叠字符如 i, j, :, !)
    boxes.sort(key=lambda b: b[0])
    merged_boxes = []
    for b in boxes:
        if not merged_boxes:
            merged_boxes.append(b)
            continue
            
        curr = merged_boxes[-1]
        nxt = b
        
        # 计算水平重叠区域宽度
        overlap = min(curr[2], nxt[2]) - max(curr[0], nxt[0])
        w1 = curr[2] - curr[0]
        w2 = nxt[2] - nxt[0]
        
        # 只有当重叠区域超过较窄部件宽度的 40% 时，才判定为同一字符部件
        if overlap > 0 and (overlap / max(1, min(w1, w2))) > 0.4:
            curr[0] = min(curr[0], nxt[0])
            curr[1] = min(curr[1], nxt[1])
            curr[2] = max(curr[2], nxt[2])
            curr[3] = max(curr[3], nxt[3])
        else:
            merged_boxes.append(nxt)

    # 3. 强制一对一拓扑对齐 (Elastic Sequence Mapping)
    target_count = len(chars_to_match)
    
    # 3.1 物理块短缺 (发生严重粘连) -> 循环劈开最宽的块
    while len(merged_boxes) < target_count:
        widest_idx = max(range(len(merged_boxes)), key=lambda i: merged_boxes[i][2] - merged_boxes[i][0])
        wb = merged_boxes[widest_idx]
        mid_x = (wb[0] + wb[2]) // 2
        
        box1 = [wb[0], wb[1], mid_x, wb[3]]
        box2 = [mid_x, wb[1], wb[2], wb[3]]
        
        merged_boxes.pop(widest_idx)
        merged_boxes.insert(widest_idx, box2)
        merged_boxes.insert(widest_idx, box1)
        
    # 3.2 物理块冗余 (噪点或非预期断裂) -> 循环熔合间距最小的邻居
    while len(merged_boxes) > target_count:
        min_gap = float('inf')
        merge_idx = 0
        for i in range(len(merged_boxes) - 1):
            gap = merged_boxes[i+1][0] - merged_boxes[i][2]
            if gap < min_gap:
                min_gap = gap
                merge_idx = i
                
        b1 = merged_boxes[merge_idx]
        b2 = merged_boxes[merge_idx+1]
        new_box = [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]
        merged_boxes.pop(merge_idx)
        merged_boxes.pop(merge_idx)
        merged_boxes.insert(merge_idx, new_box)

    # 4. Y 轴密度收敛与物理测绘
    measurements = []
    
    for i, char_char in enumerate(chars_to_match):
        bx1, by1, bx2, by2 = merged_boxes[i]
        
        # 兜底防御：防止极端切分产生零宽矩阵
        if bx1 == bx2: bx2 += 1 
        
        w_local = bx2 - bx1
        h_local = by2 - by1
        
        char_core_mask = ink_mask[by1:by2, bx1:bx2]
        h_proj = np.sum(char_core_mask, axis=1) / 255.0
        
        density_threshold = get_dynamic_density_threshold(char_char, w_local)
        valid_y_indices = np.where(h_proj >= density_threshold)[0]
        
        if len(valid_y_indices) > 0:
            core_y_offset = int(np.min(valid_y_indices))
            core_h_offset = int(np.max(valid_y_indices))
            real_y = by1 + core_y_offset
            real_h = (core_h_offset - core_y_offset) + 1
        else:
            real_y, real_h = by1, h_local
            
        bottom_y = real_y + real_h
        
        measurements.append(CharMeasurement(
            char_text=char_char,
            typography_cls=get_typography_class(char_char),
            ink_bbox=(bx1, real_y, w_local, real_h),
            ink_bottom_y=float(bottom_y),
            segmentation_confidence=0.9
        ))
        
    return measurements

# =========================================================================
# 3. 统计学分析算法
# =========================================================================
def analyze_baseline_with_ransac(measurements: List[CharMeasurement]):
    """使用 RANSAC 拟合真实排版基线，生成证据"""
    # 筛选高置信度的锚点字符
    reliable_measurements = [
        m for m in measurements 
        if m.typography_cls == TypographyClass.RELIABLE and m.segmentation_confidence > 0.8
    ]
    
    if len(reliable_measurements) < 4:
        return []  # 锚点不足，输出 INCONCLUSIVE
        
    X = np.array([m.get_center_x() for m in reliable_measurements]).reshape(-1, 1)
    y = np.array([m.ink_bottom_y for m in reliable_measurements])
    
    ransac = RANSACRegressor(residual_threshold=1.5, random_state=42)
    ransac.fit(X, y)
    
    # 动态计算测量不确定度
    inlier_mask = ransac.inlier_mask_
    if np.sum(inlier_mask) >= 2:
        inlier_std = np.std(y[inlier_mask] - ransac.predict(X[inlier_mask]))
    else:
        inlier_std = 0.0
    
    median_height = np.median([m.ink_bbox[3] for m in reliable_measurements])
    measurement_uncertainty = max(1.5, median_height * 0.03, inlier_std * 1.5)
    
    anomalies = []
    for m in measurements:
        if m.typography_cls == TypographyClass.EXCLUDED:
            continue
            
        expected_y = ransac.predict(np.array([[m.get_center_x()]]))[0]
        residual = m.ink_bottom_y - expected_y
        
        # 降笔画字符允许合理的正向偏差（下沉）
        if m.typography_cls == TypographyClass.UNCERTAIN and residual > 0:
            continue 
            
        z_score = abs(residual) / measurement_uncertainty
        
        if z_score > 3.0:
            anomalies.append({
                "char": m.char_text,
                "bbox": m.ink_bbox,
                "z_score": round(float(z_score), 2),
                "confidence": round(m.segmentation_confidence, 2)
            })
            
    return anomalies

# =========================================================================
# 4. 自动化流水线
# =========================================================================
def run_forensic_pipeline(image_path: str, output_path: str = "forensic_geometry_result.jpg"):
    print(f"📄 加载图像: {image_path}")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("❌ 图像读取失败，请检查路径。")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    print("🧠 初始化 EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    
    print("🔍 提取全图文本语义锚点...")
    results = reader.readtext(image_path)
    if not results:
        print("⚠️ 未检测到任何文本。")
        return

    total_anomalies = 0

    for item in results:
        box = item[0] 
        text = item[1]
        
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x_min, x_max = max(0, int(min(xs))), min(img_rgb.shape[1], int(max(xs)))
        y_min, y_max = max(0, int(min(ys))), min(img_rgb.shape[0], int(max(ys)))
        
        # 动态自适应外扩 (行高的 5%)，防止吞噬下一行像素
        box_height = y_max - y_min
        dynamic_pad = max(1, int(box_height * 0.05))
        
        y_min = max(0, y_min - dynamic_pad)
        y_max = min(img_rgb.shape[0], y_max + dynamic_pad)

        draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)

        roi_rgb = img_rgb[y_min:y_max, x_min:x_max]
        if roi_rgb.size == 0 or roi_rgb.shape[1] < 5:
            continue

        measurements = extract_line_geometry(roi_rgb, text)
        anomalies = analyze_baseline_with_ransac(measurements)
        anomaly_bboxes = [a["bbox"] for a in anomalies]

        for m in measurements:
            cx, cy, cw, ch = m.ink_bbox
            gx1, gy1 = x_min + cx, y_min + cy
            gx2, gy2 = gx1 + cw, gy1 + ch

            if m.ink_bbox in anomaly_bboxes:
                total_anomalies += 1
                anomaly_info = next(a for a in anomalies if a["bbox"] == m.ink_bbox)
                z_score = anomaly_info["z_score"]
                
                # 若切割置信度过低，降级告警颜色
                color = "red" if anomaly_info["confidence"] > 0.8 else "orange"
                
                draw.rectangle([gx1, gy1, gx2, gy2], outline=color, width=3)
                draw.text((gx1, max(0, gy1 - 15)), f"Z:{z_score}", fill=color)
                print(f"🚨 异常发现! 文本行 [{text}] 字符 '{m.char_text}' 基线跳变, Z-Score: {z_score} (Conf: {anomaly_info['confidence']})")
            else:
                draw.rectangle([gx1, gy1, gx2, gy2], outline="blue", width=1)

    pil_img.save(output_path)
    print(f"\n✅ 分析完成! 共发现 {total_anomalies} 处字符级几何异常。")
    print(f"🖼️ 可视化结果已保存至: {Path(output_path).resolve()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_trustlens_visual.py <图片路径>")
        sys.exit(1)
        
    run_forensic_pipeline(sys.argv[1])