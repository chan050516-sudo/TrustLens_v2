# engine/app/forensics/metadata/parsers/image_structural_parser.py
"""
图片结构深度解析器 (ImageStructuralParser)

职责：解析 JPEG/PNG 的二进制结构，提取：
- JPEG: 量化表(DQT)指纹、霍夫曼表(DHT)、APP标记段、EXIF存在性
- PNG: 数据块(Chunk)校验、IHDR/pHyS 提取
- 缩略图信息提取

所有输出均为结构化数据，不产出 Evidence（由 Analyzer 消费）。
"""
import logging
import struct
import zlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from app.core.document_ir import DocumentContext
from app.forensics.metadata.interfaces import BaseParser
from app.forensics.metadata.exceptions import ParserError

logger = logging.getLogger(__name__)


# ===== ITU-T T.81 Annex K 标准霍夫曼表 (BITS + HUFFVAL) =====
# 格式: (table_class, table_id) -> 原始字节序列
# table_class: 0=DC, 1=AC; table_id: 0=Luminance, 1=Chrominance
ANNEX_K_STANDARD_TABLES = {
    # DC Luminance (0, 0): 16 BITS + 12 HUFFVAL = 28 bytes
    (0, 0): bytes([
        0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0A, 0x0B
    ]),
    # DC Chrominance (0, 1): 16 BITS + 12 HUFFVAL = 28 bytes
    (0, 1): bytes([
        0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0A, 0x0B
    ]),
    # AC Luminance (1, 0): 16 BITS + 162 HUFFVAL = 178 bytes
    (1, 0): bytes([
        0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03,
        0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
        0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
        0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
        0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
        0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
        0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
        0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
        0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
        0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4,
        0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
        0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
        0xF9, 0xFA
    ]),
    # AC Chrominance (1, 1): 16 BITS + 162 HUFFVAL = 178 bytes
    (1, 1): bytes([
        0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04,
        0x07, 0x05, 0x04, 0x04, 0x00, 0x01, 0x02, 0x77,
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
        0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
        0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
        0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
        0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34,
        0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
        0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38,
        0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
        0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
        0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
        0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96,
        0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
        0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
        0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
        0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2,
        0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
        0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9,
        0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
        0xF9, 0xFA
    ]),
}


@dataclass
class JPEGStructure:
    """JPEG 结构信息"""
    has_jfif: bool = False          # APP0 JFIF 标记
    has_exif: bool = False          # APP1 EXIF 标记
    has_photoshop: bool = False     # APP13 Photoshop 标记
    app_segments: List[str] = field(default_factory=list)  # 存在的 APP 段列表
    dqt_tables: List[str] = field(default_factory=list)    # 量化表十六进制指纹
    dht_tables: List[Dict[str, Any]] = field(default_factory=list)  # 完整 DHT 表数据
    estimated_quality: Optional[int] = None                # 估计质量 (0-100)
    width: Optional[int] = None
    height: Optional[int] = None
    has_thumbnail: bool = False
    thumbnail_width: Optional[int] = None
    thumbnail_height: Optional[int] = None
    structural_errors: List[str] = field(default_factory=list)
    trailing_bytes: int = 0
    has_photoshop_resources: bool = False
    # ===== 第二阶段新增 =====
    encoding_type: Optional[str] = None          # "Baseline", "Extended Sequential", "Progressive"
    marker_sequence: List[str] = field(default_factory=list)  # 完整标记顺序
    dht_type: Optional[str] = None               # "standard", "optimized", "mixed"


@dataclass
class PNGStructure:
    """PNG 结构信息"""
    ihdr: Optional[Dict[str, Any]] = None      # IHDR 数据块内容
    phys: Optional[Dict[str, Any]] = None      # pHYs 物理尺寸
    has_plte: bool = False                     # 是否有调色板
    has_idat: bool = False                     # 是否有图像数据
    text_chunks: List[Dict[str, str]] = field(default_factory=list)  # tEXt/zTXt
    chunk_count: int = 0
    critical_chunks: List[str] = field(default_factory=list)  # IHDR, PLTE, IDAT, IEND
    structural_errors: List[str] = field(default_factory=list)


class ImageStructuralParser(BaseParser):
    """
    图片结构深度解析器 (BaseParser 实现)
    支持 JPEG 和 PNG 格式的二进制结构分析
    """

    def name(self) -> str:
        return "image_structural"

    def parse(self, context: DocumentContext) -> Dict[str, Any]:
        file_path = context.file_path
        if not file_path.exists():
            raise ParserError(f"File not found: {file_path}")

        mime_type = context.mime_type or ""

        result: Dict[str, Any] = {
            "image_type": None,
            "width": None,
            "height": None,
            "has_thumbnail": False,
            "thumbnail_width": None,
            "thumbnail_height": None,
            "structural_errors": [],
            "jpeg": None,
            "png": None,
        }

        try:
            if mime_type in ["image/jpeg", "image/jpg"]:
                jpeg_struct = self._parse_jpeg(file_path)
                result["image_type"] = "jpeg"
                result["width"] = jpeg_struct.width
                result["height"] = jpeg_struct.height
                result["has_thumbnail"] = jpeg_struct.has_thumbnail
                result["thumbnail_width"] = jpeg_struct.thumbnail_width
                result["thumbnail_height"] = jpeg_struct.thumbnail_height
                result["structural_errors"] = jpeg_struct.structural_errors
                result["jpeg"] = {
                    "has_jfif": jpeg_struct.has_jfif,
                    "has_exif": jpeg_struct.has_exif,
                    "has_photoshop": jpeg_struct.has_photoshop,
                    "app_segments": jpeg_struct.app_segments,
                    "dqt_tables": jpeg_struct.dqt_tables,
                    "dht_tables": jpeg_struct.dht_tables,
                    "estimated_quality": jpeg_struct.estimated_quality,
                    "encoding_type": jpeg_struct.encoding_type,
                    "marker_sequence": jpeg_struct.marker_sequence,
                    "dht_type": jpeg_struct.dht_type,
                    "trailing_bytes": jpeg_struct.trailing_bytes,
                    "has_photoshop_resources": jpeg_struct.has_photoshop_resources,
                }

            elif mime_type in ["image/png"]:
                png_struct = self._parse_png(file_path)
                result["image_type"] = "png"
                result["width"] = png_struct.ihdr.get("width") if png_struct.ihdr else None
                result["height"] = png_struct.ihdr.get("height") if png_struct.ihdr else None
                result["structural_errors"] = png_struct.structural_errors
                result["png"] = {
                    "ihdr": png_struct.ihdr,
                    "phys": png_struct.phys,
                    "has_plte": png_struct.has_plte,
                    "has_idat": png_struct.has_idat,
                    "text_chunks": png_struct.text_chunks,
                    "chunk_count": png_struct.chunk_count,
                    "critical_chunks": png_struct.critical_chunks,
                }

            else:
                # 非图片格式，返回空
                result["structural_errors"].append(f"Unsupported image format: {mime_type}")

        except Exception as e:
            logger.exception(f"ImageStructuralParser failed: {e}")
            result["structural_errors"].append(f"Parser error: {str(e)}")

        return result

    # ============== JPEG 解析 ==============

    def _parse_jpeg(self, file_path: Path) -> JPEGStructure:
        """解析 JPEG 二进制结构"""
        jpeg_struct = JPEGStructure()
        jpeg_struct.marker_sequence = ["SOI"]

        try:
            with open(file_path, "rb") as f:
                data = f.read()

            if len(data) < 2 or data[0] != 0xFF or data[1] != 0xD8:
                jpeg_struct.structural_errors.append("Invalid JPEG header (missing SOI)")
                return jpeg_struct

            pos = 2
            while pos < len(data) - 1:
                if data[pos] != 0xFF:
                    pos += 1
                    continue

                marker = data[pos + 1]
                pos += 2

                if marker == 0xD9:  # EOI
                    jpeg_struct.marker_sequence.append("EOI")
                    break
                if 0xD0 <= marker <= 0xD7 or marker == 0x01:
                    continue

                if pos + 1 >= len(data):
                    break

                segment_len = struct.unpack(">H", data[pos:pos + 2])[0]
                if segment_len < 2 or segment_len > 65535:
                    # 无效段长度，跳过
                    pos += 1
                    continue
                segment_data = data[pos + 2:pos + segment_len] if segment_len > 2 else b""
                pos += segment_len

                marker_name = self._get_marker_name(marker)
                jpeg_struct.marker_sequence.append(marker_name)

                # APP 段处理
                if 0xE0 <= marker <= 0xEF:
                    app_name = self._get_app_name(marker)
                    jpeg_struct.app_segments.append(app_name)

                    if marker == 0xE0 and segment_data.startswith(b"JFIF"):
                        jpeg_struct.has_jfif = True
                    elif marker == 0xE1 and segment_data.startswith(b"Exif"):
                        jpeg_struct.has_exif = True
                        thumb_info = self._extract_thumbnail_info(segment_data)
                        if thumb_info:
                            jpeg_struct.has_thumbnail = True
                            jpeg_struct.thumbnail_width = thumb_info.get("width")
                            jpeg_struct.thumbnail_height = thumb_info.get("height")
                    elif marker == 0xED and segment_data.startswith(b"Photoshop"):
                        jpeg_struct.has_photoshop = True
                        if segment_data.startswith(b"Photoshop 3.0"):
                            jpeg_struct.has_photoshop_resources = True

                # DQT 处理
                elif marker == 0xDB:
                    dqt_fingerprint = self._extract_dqt_fingerprint(segment_data)
                    if dqt_fingerprint:
                        jpeg_struct.dqt_tables.append(dqt_fingerprint)

                # ===== DHT 处理：存储完整表数据 =====
                elif marker == 0xC4:
                    if len(segment_data) < 17:
                        continue
                    table_class = (segment_data[0] >> 4) & 0x0F
                    table_id = segment_data[0] & 0x0F
                    bits = segment_data[1:17]
                    huffval_len = sum(bits)
                    if len(segment_data) < 17 + huffval_len:
                        continue
                    huffval = segment_data[17:17 + huffval_len]
                    table_bytes = bits + huffval
                    jpeg_struct.dht_tables.append({
                        "class": table_class,
                        "id": table_id,
                        "bits": bits,
                        "huffval": huffval,
                        "raw_bytes": table_bytes,
                    })

                # ===== SOF 处理：提取尺寸和编码类型 =====
                elif 0xC0 <= marker <= 0xCF and marker not in [0xC4, 0xC8, 0xCC]:
                    if len(segment_data) >= 7:
                        jpeg_struct.height = struct.unpack(">H", segment_data[1:3])[0]
                        jpeg_struct.width = struct.unpack(">H", segment_data[3:5])[0]
                        # 确定编码类型
                        if marker == 0xC0:
                            jpeg_struct.encoding_type = "Baseline"
                        elif marker == 0xC1:
                            jpeg_struct.encoding_type = "Extended Sequential"
                        elif marker == 0xC2:
                            jpeg_struct.encoding_type = "Progressive"
                        elif marker == 0xC3:
                            jpeg_struct.encoding_type = "Lossless"
                        elif marker in (0xC5, 0xC6, 0xC7):
                            jpeg_struct.encoding_type = "Extended Sequential (Differential)"
                        elif marker in (0xC9, 0xCA, 0xCB):
                            jpeg_struct.encoding_type = "Progressive (Differential)"
                        else:
                            jpeg_struct.encoding_type = f"SOF_{marker:x}"

                # 提取 DQT 估算质量
                if jpeg_struct.dqt_tables:
                    jpeg_struct.estimated_quality = self._estimate_quality_from_dqt(jpeg_struct.dqt_tables[0])

            # 尾部游离数据
            last_eoi = data.rfind(b'\xff\xd9')
            if last_eoi != -1:
                jpeg_struct.trailing_bytes = len(data) - (last_eoi + 2)

            # ===== DHT 类型检测 =====
            jpeg_struct.dht_type = self._classify_dht_tables(jpeg_struct.dht_tables)

        except Exception as e:
            logger.exception(f"JPEG parse error: {e}")
            jpeg_struct.structural_errors.append(f"Parse error: {str(e)}")

        return jpeg_struct


    def _get_app_name(self, marker: int) -> str:
        """获取 APP 段名称"""
        app_names = {
            0xE0: "APP0_JFIF",
            0xE1: "APP1_EXIF",
            0xE2: "APP2_ICC",
            0xE3: "APP3",
            0xE4: "APP4",
            0xE5: "APP5",
            0xE6: "APP6",
            0xE7: "APP7",
            0xE8: "APP8",
            0xE9: "APP9",
            0xEA: "APP10",
            0xEB: "APP11",
            0xEC: "APP12",
            0xED: "APP13_Photoshop",
            0xEE: "APP14_Adobe",
            0xEF: "APP15",
        }
        return app_names.get(marker, f"APP{marker-0xE0}")

    def _get_marker_name(self, marker: int) -> str:
        """返回标记的标准化名称"""
        if marker == 0xD8:
            return "SOI"
        elif marker == 0xD9:
            return "EOI"
        elif 0xE0 <= marker <= 0xEF:
            return self._get_app_name(marker)
        elif marker == 0xDB:
            return "DQT"
        elif marker == 0xC4:
            return "DHT"
        elif marker == 0xC0:
            return "SOF0"
        elif marker == 0xC1:
            return "SOF1"
        elif marker == 0xC2:
            return "SOF2"
        elif marker == 0xC3:
            return "SOF3"
        elif marker == 0xC5:
            return "SOF5"
        elif marker == 0xC6:
            return "SOF6"
        elif marker == 0xC7:
            return "SOF7"
        elif marker == 0xC9:
            return "SOF9"
        elif marker == 0xCA:
            return "SOF10"
        elif marker == 0xCB:
            return "SOF11"
        elif marker == 0xDA:
            return "SOS"
        elif marker == 0xDD:
            return "DRI"
        elif marker == 0xFE:
            return "COM"
        else:
            return f"0x{marker:02X}"

    def _extract_dqt_fingerprint(self, data: bytes) -> Optional[str]:
        """
        提取量化表指纹 (前 16 字节作为指纹)
        不同相机/软件有独特的量化值
        """
        if len(data) < 16:
            return None
        # 取前 8 个量化值 (64 是 JPEG 标准量化表大小)
        values = []
        pos = 0
        while pos < len(data) and len(values) < 8:
            # 每个量化值 1 字节 (精确度 0: 8-bit, 1: 16-bit)
            if pos + 1 >= len(data):
                break
            # 跳过表描述符 (1 byte)
            pos += 1
            # 读取量化值
            if pos >= len(data):
                break
            values.append(data[pos])
            pos += 1

        if not values:
            return None
        return "".join(f"{v:02x}" for v in values[:8])

    def _classify_dht_tables(self, dht_tables: List[Dict[str, Any]]) -> Optional[str]:
        """根据 ITU-T Annex K 标准表判定 DHT 类型"""
        if not dht_tables:
            return None

        matches = []
        for tbl in dht_tables:
            key = (tbl["class"], tbl["id"])
            standard_data = ANNEX_K_STANDARD_TABLES.get(key)
            if standard_data and tbl["raw_bytes"] == standard_data:
                matches.append(True)
            else:
                matches.append(False)

        if all(matches):
            return "standard"
        elif not any(matches):
            return "optimized"
        else:
            return "mixed"

    def _extract_dht_fingerprint(self, data: bytes) -> Optional[str]:
        """提取霍夫曼表指纹 (前 16 字节)"""
        if len(data) < 16:
            return None
        # 取前 8 个字节作为指纹
        return data[:16].hex()

    def _extract_thumbnail_info(self, exif_data: bytes) -> Optional[Dict[str, int]]:
        """
        从 EXIF 数据中提取缩略图尺寸
        简单实现：查找 JFIF 缩略图标记
        """
        # EXIF 缩略图通常以 FF D8 开始
        thumb_start = exif_data.find(b"\xFF\xD8")
        if thumb_start == -1:
            return None

        thumb_data = exif_data[thumb_start:]
        if len(thumb_data) < 10:
            return None

        # 尝试找到 SOF 标记获取尺寸 (简化为返回 None，由上层逻辑处理)
        # 实际缩略图尺寸提取较复杂，这里只标记存在性
        return {"width": 0, "height": 0}  # 标记存在，尺寸由上层使用 EXIF 元数据替代

    def _estimate_quality_from_dqt(self, dqt_fingerprint: str) -> Optional[int]:
        """
        根据量化表指纹估算 JPEG 质量
        这是一个简化的启发式估计，用于检测质量标注不一致
        """
        if not dqt_fingerprint or len(dqt_fingerprint) < 16:
            return None

        try:
            # 取前几个量化值，计算平均值
            avg = sum(int(dqt_fingerprint[i:i+2], 16) for i in range(0, 16, 2)) / 8
            if avg <= 5:
                return 95
            elif avg <= 10:
                return 85
            elif avg <= 20:
                return 70
            elif avg <= 35:
                return 55
            elif avg <= 50:
                return 40
            else:
                return 25
        except ValueError:
            return None

    # ============== PNG 解析 ==============

    def _parse_png(self, file_path: Path) -> PNGStructure:
        """解析 PNG 二进制结构"""
        png_struct = PNGStructure()

        try:
            with open(file_path, "rb") as f:
                data = f.read()

            # PNG 签名检查: 89 50 4E 47 0D 0A 1A 0A
            if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
                png_struct.structural_errors.append("Invalid PNG signature")
                return png_struct

            pos = 8
            critical_count = 0

            while pos < len(data) - 8:
                chunk_len = struct.unpack(">I", data[pos:pos+4])[0]
                chunk_type = data[pos+4:pos+8].decode("ascii", errors="ignore")
                pos += 8
                chunk_data = data[pos:pos+chunk_len] if chunk_len > 0 else b""
                pos += chunk_len

                # 跳过 CRC (4 bytes)
                pos += 4

                png_struct.chunk_count += 1

                # 关键块
                if chunk_type == "IHDR":
                    png_struct.critical_chunks.append("IHDR")
                    if len(chunk_data) >= 13:
                        png_struct.ihdr = {
                            "width": struct.unpack(">I", chunk_data[0:4])[0],
                            "height": struct.unpack(">I", chunk_data[4:8])[0],
                            "bit_depth": chunk_data[8],
                            "color_type": chunk_data[9],
                            "compression": chunk_data[10],
                            "filter": chunk_data[11],
                            "interlace": chunk_data[12],
                        }
                        critical_count += 1

                elif chunk_type == "PLTE":
                    png_struct.has_plte = True
                    png_struct.critical_chunks.append("PLTE")
                    critical_count += 1

                elif chunk_type == "IDAT":
                    png_struct.has_idat = True
                    png_struct.critical_chunks.append("IDAT")
                    critical_count += 1

                elif chunk_type == "IEND":
                    png_struct.critical_chunks.append("IEND")
                    critical_count += 1

                elif chunk_type == "pHYs":
                    if len(chunk_data) >= 9:
                        png_struct.phys = {
                            "pixels_per_unit_x": struct.unpack(">I", chunk_data[0:4])[0],
                            "pixels_per_unit_y": struct.unpack(">I", chunk_data[4:8])[0],
                            "unit": chunk_data[8],
                        }

                elif chunk_type in ["tEXt", "zTXt", "iTXt"]:
                    # 提取文本块
                    try:
                        null_pos = chunk_data.find(b"\x00")
                        if null_pos != -1:
                            keyword = chunk_data[:null_pos].decode("utf-8", errors="ignore")
                            value = chunk_data[null_pos+1:].decode("utf-8", errors="ignore")
                            png_struct.text_chunks.append({
                                "type": chunk_type,
                                "keyword": keyword,
                                "value": value[:200]  # 截断长文本
                            })
                    except Exception:
                        pass

            # 检查关键块完整性
            required = ["IHDR", "IDAT", "IEND"]
            missing = [r for r in required if r not in png_struct.critical_chunks]
            if missing:
                png_struct.structural_errors.append(f"Missing critical chunks: {missing}")

        except Exception as e:
            logger.exception(f"PNG parse error: {e}")
            png_struct.structural_errors.append(f"Parse error: {str(e)}")

        return png_struct