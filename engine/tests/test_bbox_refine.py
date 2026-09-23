#!/usr/bin/env python3
# engine/tests/test_bbox_refine.py
"""
ImageObservationExtractor bbox 收紧效果可视化测试

用法:
    python tests/test_bbox_refine.py                       # 默认用 test_doc/bank_statement_image_test_4.jpg
    python tests/test_bbox_refine.py <image_path>
    python tests/test_bbox_refine.py <image_path> --verbose
    python tests/test_bbox_refine.py <image_path> --out-dir <dir>
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.document_ir import DocumentContext
from app.perception.extractors.image_observation_extractor import (
    ImageObservationExtractor,
)
from app.perception.models.bbox import BBox


# 颜色 (BGR)
COLOR_RAW = (0, 200, 0)          # 绿 - OCR 原始 bbox
COLOR_REFINED = (0, 0, 255)      # 红 - 收紧后 bbox
COLOR_REFINED_OK = (0, 165, 255) # 橙 - refine 成功但变化不大


def draw_bbox(img, bbox: BBox, color, thickness=2):
    p1 = (int(round(bbox.x0)), int(round(bbox.y0)))
    p2 = (int(round(bbox.x1)), int(round(bbox.y1)))
    cv2.rectangle(img, p1, p2, color, thickness)


def draw_label(img, bbox: BBox, text: str, color):
    x = int(round(bbox.x0))
    y = int(round(bbox.y0)) - 4
    if y < 12:
        y = int(round(bbox.y1)) + 14
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
    )


def area(b: BBox) -> float:
    return max(0.0, b.width) * max(0.0, b.height)


def main():
    parser = argparse.ArgumentParser()
    default_img = PROJECT_ROOT / "test_doc" / "bank_statement_image_test_4.jpg"
    parser.add_argument("file", nargs="?", default=str(default_img))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else PROJECT_ROOT / "tests" / "test_results" / file_path.stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"ImageObservationExtractor bbox 收紧测试")
    print(f"{'='*72}")
    print(f"  输入  : {file_path}")
    print(f"  输出  : {out_dir}")

    # ---------- 1. 跑 OCR（不 refine）----------
    context = DocumentContext(file_path=file_path, mime_type="image/jpeg")
    extractor = ImageObservationExtractor(use_bbox_refinement=False)

    print(f"\n[1] Running OCR (without refinement)...")
    obs_raw = extractor.extract(context)
    print(f"    → {len(obs_raw)} observations")

    # 加载图片（用于手动 refine）
    image_bgr = extractor._load_image(file_path)
    if image_bgr is None:
        print(f"❌ Failed to load image")
        sys.exit(1)
    print(f"    Image size: {image_bgr.shape[1]}x{image_bgr.shape[0]}")

    # ---------- 2. 对每个 obs 手动 refine ----------
    print(f"\n[2] Applying ink refinement to each observation...")
    refined_list: List[Optional[BBox]] = []
    for obs in obs_raw:
        refined = extractor._refine_bbox_with_ink(image_bgr, obs.bbox)
        refined_list.append(refined)

    n_refined_ok = sum(1 for r in refined_list if r is not None)
    print(f"    → {n_refined_ok}/{len(obs_raw)} refined successfully")

    # ---------- 3. 打印对比表 ----------
    print(f"\n{'='*72}")
    print(f"bbox comparison (raw vs refined)")
    print(f"{'='*72}")
    print(
        f"  {'#':>3}  "
        f"{'raw (x0,y0,x1,y1)':<28}  "
        f"{'refined (x0,y0,x1,y1)':<28}  "
        f"{'Δarea':>8}"
    )
    print(f"  {'-'*3}  {'-'*28}  {'-'*28}  {'-'*8}")

    ratios: List[float] = []
    for i, (obs, refined) in enumerate(zip(obs_raw, refined_list)):
        raw = obs.bbox
        raw_s = f"({raw.x0:.0f},{raw.y0:.0f},{raw.x1:.0f},{raw.y1:.0f})"
        if refined is None:
            ref_s = "(none)"
            ratio = 1.0
        else:
            ref_s = f"({refined.x0:.0f},{refined.y0:.0f},{refined.x1:.0f},{refined.y1:.0f})"
            a_raw = area(raw)
            a_ref = area(refined)
            ratio = a_ref / a_raw if a_raw > 0 else 1.0
            ratios.append(ratio)

        if args.verbose or i < 20:
            print(f"  {i:>3}  {raw_s:<28}  {ref_s:<28}  {ratio*100:>6.1f}%")

    if not args.verbose and len(obs_raw) > 20:
        print(f"  ... and {len(obs_raw) - 20} more")

    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        print(f"\n  平均面积比 (refined / raw): {avg_ratio*100:.1f}%")
        print(f"  > 100% 表示 refine 后反而变大 (OCR 漏了部分墨迹)")
        print(f"  < 100% 表示成功收紧")

    # ---------- 4. 可视化 ----------
    print(f"\n[3] Generating visualizations...")

    # 4.1 raw
    img_raw = image_bgr.copy()
    for obs in obs_raw:
        draw_bbox(img_raw, obs.bbox, COLOR_RAW, thickness=1)
    cv2.imwrite(str(out_dir / "01_raw_bboxes.jpg"), img_raw)

    # 4.2 refined
    img_ref = image_bgr.copy()
    for i, refined in enumerate(refined_list):
        if refined is None:
            continue
        draw_bbox(img_ref, refined, COLOR_REFINED, thickness=1)
    cv2.imwrite(str(out_dir / "02_refined_bboxes.jpg"), img_ref)

    # 4.3 对比（raw 绿细线 + refined 红粗线）
    img_cmp = image_bgr.copy()
    for obs, refined in zip(obs_raw, refined_list):
        draw_bbox(img_cmp, obs.bbox, COLOR_RAW, thickness=1)
        if refined is not None:
            draw_bbox(img_cmp, refined, COLOR_REFINED, thickness=2)
    cv2.imwrite(str(out_dir / "03_compare.jpg"), img_cmp)

    # 4.4 大图局部放大（上半部分 + 下半部分各一张，看得更清楚）
    h, w = image_bgr.shape[:2]
    mid = h // 2
    if h > 1200:
        top = img_cmp[0:mid, :]
        bot = img_cmp[mid:h, :]
        cv2.imwrite(str(out_dir / "03a_compare_top.jpg"), top)
        cv2.imwrite(str(out_dir / "03b_compare_bottom.jpg"), bot)

    # 4.5 少数样本的 ROI 并排展示
    n_sample = min(8, len(obs_raw))
    if n_sample > 0:
        rows = []
        for i in range(n_sample):
            obs = obs_raw[i]
            refined = refined_list[i]

            raw = obs.bbox
            # 取 raw bbox 外扩一点
            pad = 6
            rx0 = max(0, int(raw.x0) - pad)
            ry0 = max(0, int(raw.y0) - pad)
            rx1 = min(w, int(raw.x1) + pad)
            ry1 = min(h, int(raw.y1) + pad)
            roi = image_bgr[ry0:ry1, rx0:rx1].copy()

            if roi.size == 0:
                continue

            # 画 raw bbox
            cv2.rectangle(
                roi,
                (int(raw.x0) - rx0, int(raw.y0) - ry0),
                (int(raw.x1) - rx0, int(raw.y1) - ry0),
                COLOR_RAW, 1,
            )
            if refined is not None:
                cv2.rectangle(
                    roi,
                    (int(refined.x0) - rx0, int(refined.y0) - ry0),
                    (int(refined.x1) - rx0, int(refined.y1) - ry0),
                    COLOR_REFINED, 2,
                )

            # 放大 3 倍方便观察
            scale = 3
            big = cv2.resize(
                roi, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )
            rows.append((i, obs.text, big))

        if rows:
            # 简单纵向拼接
            max_w = max(r[2].shape[1] for r in rows)
            total_h = sum(r[2].shape[0] + 30 for r in rows)
            canvas = np.full((total_h, max_w + 20, 3), 255, dtype=np.uint8)
            y_cursor = 0
            for idx, text, img in rows:
                canvas[y_cursor:y_cursor+20, :] = 255
                cv2.putText(
                    canvas,
                    f"#{idx}  '{text[:60]}'",
                    (10, y_cursor + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
                )
                y_cursor += 20
                hh, ww = img.shape[:2]
                canvas[y_cursor:y_cursor+hh, 10:10+ww] = img
                y_cursor += hh + 10
            cv2.imwrite(str(out_dir / "04_samples.jpg"), canvas)

    print(f"\n  ✅ Saved:")
    for f in sorted(out_dir.glob("*.jpg")):
        print(f"     {f.name}  ({f.stat().st_size:,} bytes)")

    print(f"\n  颜色约定:")
    print(f"    绿细线 = OCR 原始 bbox")
    print(f"    红粗线 = refine 后 bbox")
    print(f"    橙色   = 保留备用")

    print(f"\n{'='*72}")
    print(f"Done. Output: {out_dir}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()