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

class InferenceFlag(IntEnum):
    OBSERVED = 0          # 物理墨迹自然切分 (高可信)
    SPLIT_INFERRED = 1    # 因粘连被算法强制劈开 (降级)
    MERGED_INFERRED = 2   # 因破碎被算法强制熔合 (降级)

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
    inference_flag: InferenceFlag        # 法证推断标记
    
    ink_bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    ink_bottom_y: float                  # 重命名：明确这是墨迹最低点而非基线
    
    global_line_confidence: float        # 整行的宏观切割质量
    local_char_confidence: float         # 单字符的微观切割质量
    
    def get_center_x(self):
        return self.ink_bbox[0] + self.ink_bbox[2] / 2.0
    
    @property
    def final_confidence(self):
        return self.global_line_confidence * self.local_char_confidence

# =========================================================================
# 2. 形态学除线与几何提取 (吸收 Point 11, 5, 6)
# =========================================================================
def remove_structural_lines(binary_mask: np.ndarray) -> np.ndarray:
    """剔除表格横线与下划线，防止字符串连 (Point 11)"""
    # 假设横线至少有全图宽度的 1/4 长，高度极窄
    kernel_length = max(20, binary_mask.shape[1] // 4)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    
    # 提取出长横线
    detected_lines = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, horizontal_kernel)
    # 从原掩码中减去横线
    clean_mask = cv2.subtract(binary_mask, detected_lines)
    return clean_mask

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
    gray = cv2.cvtColor(line_rgb, cv2.COLOR_RGB2GRAY)
    ink_mask = robust_ink_segmentation(gray)  # 调用你已有的二值化函数
    ink_mask = remove_structural_lines(ink_mask) # 插入除线逻辑
    
    chars_to_match = [c for c in ocr_text if not c.isspace()]
    if not chars_to_match:
        return []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    boxes = [[x, y, x + w, y + h] for i, (x, y, w, h, area) in enumerate(stats) if i > 0 and w >= 1 and h >= 2]
            
    if not boxes:
        return []

    # X轴深度重叠合并 (处理 i, j, :)
    boxes.sort(key=lambda b: b[0])
    merged_boxes = []
    for b in boxes:
        if not merged_boxes:
            merged_boxes.append(b)
            continue
        curr = merged_boxes[-1]
        nxt = b
        overlap = min(curr[2], nxt[2]) - max(curr[0], nxt[0])
        w1, w2 = curr[2] - curr[0], nxt[2] - nxt[0]
        if overlap > 0 and (overlap / max(1, min(w1, w2))) > 0.4:
            curr[0], curr[1] = min(curr[0], nxt[0]), min(curr[1], nxt[1])
            curr[2], curr[3] = max(curr[2], nxt[2]), max(curr[3], nxt[3])
        else:
            merged_boxes.append(nxt)

    target_count = len(chars_to_match)
    # 初始化全局置信度与推断标记池
    global_line_conf = max(0.0, 1.0 - abs(len(merged_boxes) - target_count) / max(1, target_count))
    box_flags = [InferenceFlag.OBSERVED] * len(merged_boxes)

    # 3.1 强制劈开 (Point 5)
    while len(merged_boxes) < target_count:
        widest_idx = max(range(len(merged_boxes)), key=lambda i: merged_boxes[i][2] - merged_boxes[i][0])
        wb = merged_boxes[widest_idx]
        mid_x = (wb[0] + wb[2]) // 2
        
        merged_boxes.pop(widest_idx)
        box_flags.pop(widest_idx)
        
        merged_boxes.insert(widest_idx, [mid_x, wb[1], wb[2], wb[3]])
        merged_boxes.insert(widest_idx, [wb[0], wb[1], mid_x, wb[3]])
        # 标记为推断切分
        box_flags.insert(widest_idx, InferenceFlag.SPLIT_INFERRED)
        box_flags.insert(widest_idx, InferenceFlag.SPLIT_INFERRED)
        
    # 3.2 强制熔合 (Point 6)
    while len(merged_boxes) > target_count:
        min_gap, merge_idx = float('inf'), 0
        for i in range(len(merged_boxes) - 1):
            gap = merged_boxes[i+1][0] - merged_boxes[i][2]
            if gap < min_gap:
                min_gap, merge_idx = gap, i
                
        b1, b2 = merged_boxes[merge_idx], merged_boxes[merge_idx+1]
        new_box = [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]
        
        merged_boxes.pop(merge_idx)
        merged_boxes.pop(merge_idx)
        box_flags.pop(merge_idx)
        box_flags.pop(merge_idx)
        
        merged_boxes.insert(merge_idx, new_box)
        box_flags.insert(merge_idx, InferenceFlag.MERGED_INFERRED)

    measurements = []
    for i, char_char in enumerate(chars_to_match):
        bx1, by1, bx2, by2 = merged_boxes[i]
        flag = box_flags[i]
        
        # 局部置信度：如果是算法强行算出来的框，局部置信度大打折扣
        local_conf = 1.0 if flag == InferenceFlag.OBSERVED else 0.4
        
        if bx1 == bx2: bx2 += 1 
        w_local, h_local = bx2 - bx1, by2 - by1
        
        char_core_mask = ink_mask[by1:by2, bx1:bx2]
        h_proj = np.sum(char_core_mask, axis=1) / 255.0
        
        density_threshold = get_dynamic_density_threshold(char_char, w_local) # 调用已有函数
        valid_y_indices = np.where(h_proj >= density_threshold)[0]
        
        if len(valid_y_indices) > 0:
            core_y_offset = int(np.min(valid_y_indices))
            core_h_offset = int(np.max(valid_y_indices))
            real_y = by1 + core_y_offset
            real_h = (core_h_offset - core_y_offset) + 1
        else:
            real_y, real_h = by1, h_local
            
        measurements.append(CharMeasurement(
            char_text=char_char,
            typography_cls=get_typography_class(char_char),
            inference_flag=flag,
            ink_bbox=(bx1, real_y, w_local, real_h),
            ink_bottom_y=float(real_y + real_h),
            global_line_confidence=global_line_conf,
            local_char_confidence=local_conf
        ))
        
    return measurements

# =========================================================================
# 3. 统计学分析算法
# =========================================================================
def analyze_baseline_with_ransac(measurements: List[CharMeasurement]):
    # 筛选高置信度的锚点字符 (仅限 OBSERVED 且属于 RELIABLE)
    reliable_anchors = [
        m for m in measurements 
        if m.typography_cls == TypographyClass.RELIABLE and m.final_confidence > 0.8
    ]
    
    if len(reliable_anchors) < 4:
        return []  
        
    X = np.array([m.get_center_x() for m in reliable_anchors]).reshape(-1, 1)
    y = np.array([m.ink_bottom_y for m in reliable_anchors])
    
    # 1. RANSAC Inlier 阈值：用于模型拟合，定义“排版公差” (通常较小，如字高的 5%)
    median_height = np.median([m.ink_bbox[3] for m in reliable_anchors])
    ransac_threshold = max(1.0, median_height * 0.05)
    
    ransac = RANSACRegressor(residual_threshold=ransac_threshold, random_state=42)
    ransac.fit(X, y)
    
    inlier_mask = ransac.inlier_mask_
    inlier_std = np.std(y[inlier_mask] - ransac.predict(X[inlier_mask])) if np.sum(inlier_mask) >= 2 else 0.0
    
    # 2. 测量不确定度 (Measurement Uncertainty)：决定报告异常的底线
    # 结合图像分辨率下限与实际内点噪声，作为异常判定的统计分母
    measurement_uncertainty = max(1.5, median_height * 0.03, inlier_std * 2.0)
    
    anomalies = []
    for m in measurements:
        if m.typography_cls == TypographyClass.EXCLUDED:
            continue
            
        expected_y = ransac.predict(np.array([[m.get_center_x()]]))[0]
        residual = m.ink_bottom_y - expected_y
        
        if m.typography_cls == TypographyClass.UNCERTAIN and residual > 0:
            continue 
            
        z_score = abs(residual) / measurement_uncertainty
        
        if z_score > 2.0:
            anomalies.append({
                "char": m.char_text,
                "bbox": m.ink_bbox,
                "z_score": round(float(z_score), 2),
                "confidence": round(m.final_confidence, 2),
                "inference_type": m.inference_flag.name
            })
            
    return anomalies

# =========================================================================
# 4. 自动化流水线
# =========================================================================
def run_forensic_pipeline(image_path: str, output_path: str = "forensic_geometry_result2.jpg"):
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