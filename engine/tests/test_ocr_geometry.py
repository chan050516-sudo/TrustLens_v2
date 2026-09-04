import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw


# ==========================================
# 你的核心形态学与几何测量算法
# ==========================================
@dataclass
class CharMeasurement:
  char_text: str
  bbox: Tuple[int, int, int, int]
  baseline_y: float
  is_descender: bool


def robust_ink_segmentation(gray_line: np.ndarray) -> np.ndarray:
    """高保真墨迹分割：保留抗锯齿边缘"""
    # 1. 局部对比度拉伸，避免灰色的抗锯齿边缘被判定为背景
    p2, p98 = np.percentile(gray_line, (2, 98))
    img_rescale = np.clip((gray_line - p2) / (p98 - p2 + 1e-5) * 255, 0, 255).astype(np.uint8)

    bg = cv2.GaussianBlur(img_rescale, (31, 31), 0)
    diff = cv2.subtract(bg, img_rescale)
    
    # 2. 使用 THRESH_TRIANGLE 替代 OTSU，它对“大面积背景+微小文字边缘”的单峰直方图切分更准，能包住更多边缘像素
    _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    
    # ⚠️ 绝对禁止使用 cv2.morphologyEx(MORPH_OPEN)，保留原生的像素边缘
    return binary

def extract_line_geometry(line_rgb: np.ndarray, ocr_text: str) -> List[CharMeasurement]:
    """带有二次像素收缩约束的物理几何提取"""
    gray = cv2.cvtColor(line_rgb, cv2.COLOR_RGB2GRAY)
    ink_mask = robust_ink_segmentation(gray)
    h_img, w_img = ink_mask.shape

    # 1. 垂直投影切分大致区间
    v_proj = np.sum(ink_mask, axis=0) / 255.0
    in_char = False
    segments = []
    start_x = 0
    
    for x in range(w_img):
        if v_proj[x] > 0 and not in_char:
            in_char = True
            start_x = x
        elif v_proj[x] == 0 and in_char:
            in_char = False
            if x - start_x >= 1: 
                segments.append((start_x, x))
    if in_char:
        segments.append((start_x, w_img))

    DESCENDERS = set("gjpqyQ,")
    measurements = []
    chars_to_match = [c for c in ocr_text if not c.isspace()]
    num_matches = min(len(segments), len(chars_to_match))

    for i in range(num_matches):
        sx, ex = segments[i]
        char_char = chars_to_match[i]
        
        char_mask = ink_mask[:, sx:ex]
        coords = cv2.findNonZero(char_mask)
        if coords is None:
            continue
            
        x_local, y_local, w_local, h_local = cv2.boundingRect(coords)
        
        # --- 核心修正：Y轴抗噪密度收敛 (Robust Y-Bounds) ---
        # 截取该字符的紧凑掩码区
        char_core_mask = char_mask[y_local:y_local+h_local, x_local:x_local+w_local]
        # 计算水平方向（每一行）的墨迹像素点个数
        h_proj = np.sum(char_core_mask, axis=1) / 255.0
        
        # 设定阈值：该行的墨迹宽度必须大于整个字符宽度的 15%（或者是至少 2 个像素）
        # 这一步直接杀掉贴图边缘稀疏的 1~2 像素伪影
        density_threshold = max(2.0, w_local * 0.15)
        valid_y_indices = np.where(h_proj >= density_threshold)[0]
        
        if len(valid_y_indices) > 0:
            core_y_offset = int(np.min(valid_y_indices))
            core_h_offset = int(np.max(valid_y_indices))
            
            real_y = y_local + core_y_offset
            real_h = (core_h_offset - core_y_offset) + 1
        else:
            real_y = y_local
            real_h = h_local
            
        real_x = sx + x_local
        real_w = w_local
        bottom_y = real_y + real_h
        
        measurements.append(CharMeasurement(
            char_text=char_char,
            bbox=(real_x, real_y, real_w, real_h),
            baseline_y=float(bottom_y),
            is_descender=(char_char in DESCENDERS)
        ))

    return measurements


def analyze_baseline_discrepancy(measurements: List[CharMeasurement]):
  baseline_candidates = [
      m.baseline_y for m in measurements if not m.is_descender
  ]
  if len(baseline_candidates) < 3:
    return []

  median_base = np.median(baseline_candidates)
  mad = np.median(np.abs(baseline_candidates - median_base))
  
  # 获取该行非降笔画字符的中位高度
  heights = [m.bbox[3] for m in measurements if not m.is_descender]
  median_height = np.median(heights) if heights else 10.0
  
  # 动态排版容差：字高的 3%，且受限于像素栅格化，设定物理极限为 1.5 像素
  dynamic_min_std = max(1.5, median_height * 0.03)
  
  # 取统计离散度与动态物理容差的较大值
  robust_std = max(1.4826 * mad, dynamic_min_std)

  anomalies = []
  for m in measurements:
    if m.is_descender:
      continue
    z_score = abs(m.baseline_y - median_base) / robust_std
    if z_score > 3.0:
      anomalies.append({
          "char": m.char_text,
          "bbox": m.bbox,
          "baseline_residual": m.baseline_y - median_base,
          "z_score": round(float(z_score), 2),
      })
  return anomalies


# ==========================================
# 自动化执行流水线
# ==========================================
def run_forensic_pipeline(
    image_path: str, output_path: str = "forensic_geometry_result.jpg"
):
  print(f"📄 加载图像: {image_path}")
  img_bgr = cv2.imread(image_path)
  if img_bgr is None:
    print("❌ 图像读取失败，请检查路径。")
    return

  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  pil_img = Image.fromarray(img_rgb)
  draw = ImageDraw.Draw(pil_img)

  print("🧠 初始化 EasyOCR (稳定纯 CPU 模式)...")
  reader = easyocr.Reader(["en"], gpu=False)

  print("🔍 提取全图文本语义锚点...")
  results = reader.readtext(image_path)

  if not results:
    print("⚠️ 未检测到任何文本。")
    return

  total_anomalies = 0

  for item in results:
    box = item[0]  # [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    text = item[1]

    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x_min, x_max = max(0, int(min(xs))), min(img_rgb.shape[1], int(max(xs)))
    y_min, y_max = max(0, int(min(ys))), min(img_rgb.shape[0], int(max(ys)))

    # 动态自适应外扩 (行高的 5%)
    box_height = y_max - y_min
    dynamic_pad = max(1, int(box_height * 0.05)) # 至少保留 1 像素缓冲
    
    y_min = max(0, y_min - dynamic_pad)
    y_max = min(img_rgb.shape[0], y_max + dynamic_pad)

    draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)

    roi_rgb = img_rgb[y_min:y_max, x_min:x_max]
    if roi_rgb.size == 0 or roi_rgb.shape[1] < 5:
      continue

    measurements = extract_line_geometry(roi_rgb, text)
    anomalies = analyze_baseline_discrepancy(measurements)
    anomaly_bboxes = [a["bbox"] for a in anomalies]

    for m in measurements:
      cx, cy, cw, ch = m.bbox
      gx1, gy1 = x_min + cx, y_min + cy
      gx2, gy2 = gx1 + cw, gy1 + ch

      if m.bbox in anomaly_bboxes:
        total_anomalies += 1
        anomaly_info = next(a for a in anomalies if a["bbox"] == m.bbox)
        z_score = anomaly_info["z_score"]

        draw.rectangle([gx1, gy1, gx2, gy2], outline="red", width=3)
        draw.text((gx1, max(0, gy1 - 15)), f"Z:{z_score}", fill="red")
        print(
            f"🚨 异常发现! 文本行 [{text}] 中的字符 '{m.char_text}' 基线跳变,"
            f" Z-Score: {z_score}"
        )
      else:
        draw.rectangle([gx1, gy1, gx2, gy2], outline="blue", width=1)

  pil_img.save(output_path)
  print(f"\n✅ 分析完成! 共发现 {total_anomalies} 处字符级几何异常。")
  print(f"🖼️ 可视化结果已保存至: {Path(output_path).resolve()}")


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("用法: python test_easyocr.py <图片路径>")
    sys.exit(1)

  run_forensic_pipeline(sys.argv[1])