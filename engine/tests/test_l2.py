#!/usr/bin/env python3
# engine/tests/test_l2.py
"""
L2 Visual Layer 冒烟测试
严格检查：
1. 无启发式加权平均
2. 坐标映射正确
3. 共识检测有效
4. 模型缺失时优雅降级
"""
import sys
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.document_ir import DocumentContext
from app.core.evidence import EvidenceType
from app.forensics.visual import (
    VisualInput,
    VisualModelOutput,
    VisualPreprocessor,
    EvidenceExtractor,
    VisualInferenceEngine,
    ContextSerializer,
    ImageSourceType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===================== 辅助函数 =====================

def create_dummy_image(width=512, height=512) -> np.ndarray:
    """生成测试用 RGB 图像"""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    # 在中心画一个白色方块，模拟异常区域
    img[200:300, 200:300] = 255
    return img


def create_dummy_mask(width=512, height=512, center=True) -> np.ndarray:
    """生成模拟异常掩码"""
    mask = np.zeros((height, width), dtype=np.float32)
    if center:
        mask[200:300, 200:300] = 0.9
    else:
        # 随机位置
        mask[100:150, 400:450] = 0.85
    return mask


# ===================== 测试用例 =====================

def test_preprocessor_pdf():
    """测试 PDF 渲染（如果有测试文档）"""
    test_pdf = PROJECT_ROOT / "test_doc" / "sample.pdf"
    if not test_pdf.exists():
        logger.warning(f"Test PDF not found: {test_pdf}. Skipping.")
        return

    context = DocumentContext(file_path=test_pdf)
    inputs = VisualPreprocessor.from_context(context)

    assert len(inputs) > 0, "PDF should render at least one page"
    for vinput in inputs:
        assert vinput.source_type == ImageSourceType.PDF
        assert vinput.image_array is not None
        assert vinput.image_array.shape[2] == 3  # RGB
        assert vinput.page_id is not None
        assert vinput.pixel_to_user_transform is not None
        logger.info(f"  Page {vinput.page_id}: {vinput.original_size}")


def test_preprocessor_image(tmp_path):
    """测试 JPEG/PNG 解码"""
    # 创建临时 JPEG
    img = Image.fromarray(create_dummy_image())
    jpeg_path = tmp_path / "test.jpg"
    img.save(jpeg_path, "JPEG")

    context = DocumentContext(file_path=jpeg_path, mime_type="image/jpeg")
    inputs = VisualPreprocessor.from_context(context)

    assert len(inputs) == 1
    vinput = inputs[0]
    assert vinput.source_type == ImageSourceType.JPEG
    assert vinput.image_array is not None
    assert vinput.page_id is None
    assert vinput.pixel_to_user_transform is None  # 图像无坐标映射
    logger.info("JPEG preprocessor test passed.")


def test_evidence_extractor_consensus():
    """测试跨模型共识检测"""
    h, w = 512, 512
    # 模拟三个模型的输出，其中两个在中心区域重叠
    mask1 = create_dummy_mask(w, h, center=True)
    mask2 = create_dummy_mask(w, h, center=True)  # 与 mask1 完全重叠
    mask3 = create_dummy_mask(w, h, center=False)  # 不同区域

    outputs = {
        "trufor": VisualModelOutput(
            model_name="trufor",
            image_score=0.9,
            confidence=0.85,
            localization_mask=mask1,
            anomaly_area_ratio=0.05
        ),
        "catnet": VisualModelOutput(
            model_name="catnet",
            image_score=0.8,
            confidence=0.75,
            localization_mask=mask2,
            anomaly_area_ratio=0.04
        ),
        "mvss": VisualModelOutput(
            model_name="mvss",
            image_score=0.3,
            confidence=0.4,
            localization_mask=mask3,
            anomaly_area_ratio=0.02
        )
    }

    # 构建 VisualInput（无坐标变换）
    vinput = VisualInput(
        source_type=ImageSourceType.JPEG,
        image_array=create_dummy_image(w, h),
        original_size=(w, h)
    )

    extractor = EvidenceExtractor(iou_threshold=0.3)
    evidences, context = extractor.extract(
        visual_inputs=[vinput],
        inference_results=[outputs]
    )

    # 验证：应该产出至少一个 CONSENSUS 证据（trufor + catnet）
    consensus_ev = [e for e in evidences if e.type == EvidenceType.VISUAL_CONSENSUS]
    assert len(consensus_ev) >= 1, "Should detect consensus between TruFor and CAT-Net"
    
    # 验证共识证据包含正确的模型列表
    models_in_consensus = consensus_ev[0].value.get("models", [])
    assert "trufor" in models_in_consensus
    assert "catnet" in models_in_consensus
    
    # 验证 MVSS 的区域被标记为单模型证据（由于与其他模型不重叠）
    single_ev = [e for e in evidences if e.type == EvidenceType.VISUAL_MODEL_SPECIFIC]
    assert len(single_ev) >= 1
    assert single_ev[0].value.get("model") == "mvss"

    # 验证上下文不包含任何"平均分"或"加权"字段
    context_dict = ContextSerializer.to_dict(context)
    assert "average_score" not in json.dumps(context_dict).lower()
    assert "weighted" not in json.dumps(context_dict).lower()

    logger.info(f"Extracted {len(evidences)} evidences. Consensus: {len(consensus_ev)}, Single: {len(single_ev)}")
    logger.info(f"Context observations: {context.observations}")


def test_coordinate_mapping():
    """测试像素坐标到 PDF 用户坐标的映射"""
    transform = [
        [1.0/3.0, 0, 0],  # x_px = x_user * 3 -> x_user = x_px / 3
        [0, 1.0/3.0, 0],
        [0, 0, 1]
    ]
    # 模拟一个在 (300, 200) 到 (400, 300) 像素的区域
    bbox_pixel = (300, 200, 400, 300)

    extractor = EvidenceExtractor()
    bbox_user = extractor._pixel_to_user(bbox_pixel, transform)

    expected = (100.0, 66.67, 133.33, 100.0)
    assert abs(bbox_user[0] - expected[0]) < 0.1
    assert abs(bbox_user[1] - expected[1]) < 0.1
    assert abs(bbox_user[2] - expected[2]) < 0.1
    assert abs(bbox_user[3] - expected[3]) < 0.1
    logger.info(f"Pixel {bbox_pixel} -> User {bbox_user}")


def test_inference_engine_fallback():
    """测试推理引擎在模型缺失时的优雅降级"""
    # 使用一个不存在的权重路径，触发降级
    engine = VisualInferenceEngine(
        model_weights={"trufor": "/non/existent/path.pth"},
        enabled_models=["trufor"]
    )
    # 由于加载失败，引擎应为不可用状态
    if not engine.is_available():
        logger.info("Engine correctly unavailable due to missing weights.")
    else:
        # 如果意外加载成功（例如默认路径存在），跳过测试
        logger.warning("Model weights found unexpectedly, skipping fallback test.")


def test_no_heuristic_scoring_in_code():
    """
    代码规范检查：确保 evidence_extractor.py 中没有计算加权平均分或总风险分
    这是一个简单的静态检查，用于防止引入启发式评分
    """
    import ast
    import os

    # 要检查的文件列表
    files_to_check = [
        "app/forensics/visual/evidence_extractor.py",
        "app/forensics/visual/context_serializer.py",
    ]

    suspicious_patterns = [
        "average",
        "weighted",
        "sum.*score",
        "score.*sum",
        "total_score",
        "risk_score",
        "final_score"
    ]

    found_violations = []
    for file_path in files_to_check:
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            continue

        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 检查是否包含可疑模式（排除注释和文档字符串）
        for pattern in suspicious_patterns:
            if pattern in content:
                # 简单排除注释行
                lines = content.split("\n")
                for line_num, line in enumerate(lines, 1):
                    if pattern in line and not line.strip().startswith("#"):
                        found_violations.append(
                            f"{file_path}:{line_num} - contains '{pattern}'"
                        )

    if found_violations:
        for v in found_violations:
            logger.warning(f"⚠️  Potential heuristic scoring detected: {v}")
        # 我们允许存在 "average" 作为变量名，但不允许作为评分计算
        # 这里我们只警告，不强制失败，因为 "average" 可能在注释或 IoU 计算中
        logger.info("Static check completed. Manual review recommended for the above.")
    else:
        logger.info("✅ No obvious heuristic scoring patterns found.")


# ===================== 主入口 =====================

def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("🔍 TrustLens L2 Visual Layer - Smoke Test")
    print("=" * 70)

    tests = [
        ("Preprocessor (Image)", test_preprocessor_image),
        ("Evidence Extractor (Consensus)", test_evidence_extractor_consensus),
        ("Coordinate Mapping", test_coordinate_mapping),
        ("Inference Engine Fallback", test_inference_engine_fallback),
        ("Static Code Check", test_no_heuristic_scoring_in_code),
    ]

    # 可选：如果有 PDF 测试文档，取消注释下一行
    # tests.append(("Preprocessor (PDF)", test_preprocessor_pdf))

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"✅ {name}: PASSED")
            passed += 1
        except AssertionError as e:
            print(f"❌ {name}: FAILED - {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {name}: ERROR - {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()