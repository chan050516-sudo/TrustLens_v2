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


@dataclass
class JPEGStructure:
    """JPEG 结构信息"""
    has_jfif: bool = False          # APP0 JFIF 标记
    has_exif: bool = False          # APP1 EXIF 标记
    has_photoshop: bool = False     # APP13 Photoshop 标记
    app_segments: List[str] = field(default_factory=list)  # 存在的 APP 段列表
    dqt_tables: List[str] = field(default_factory=list)    # 量化表十六进制指纹
    dht_tables: List[str] = field(default_factory=list)    # 霍夫曼表十六进制指纹
    estimated_quality: Optional[int] = None                # 估计质量 (0-100)
    width: Optional[int] = None
    height: Optional[int] = None
    has_thumbnail: bool = False
    thumbnail_width: Optional[int] = None
    thumbnail_height: Optional[int] = None
    structural_errors: List[str] = field(default_factory=list)


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
        struct = JPEGStructure()

        try:
            with open(file_path, "rb") as f:
                data = f.read()

            # JPEG 必须以 FF D8 开头
            if len(data) < 2 or data[0] != 0xFF or data[1] != 0xD8:
                struct.structural_errors.append("Invalid JPEG header (missing SOI)")
                return struct

            # 遍历 JPEG 标记段
            pos = 2  # 跳过 SOI
            while pos < len(data) - 1:
                if data[pos] != 0xFF:
                    # 可能是填充字节，跳过
                    pos += 1
                    continue

                marker = data[pos + 1]
                pos += 2

                # 标记类型: 0xD0-0xD7 = RST (重启), 0x01 = TEM, 0xD9 = EOI
                if marker == 0xD9:  # EOI (End of Image)
                    break
                if 0xD0 <= marker <= 0xD7 or marker == 0x01:
                    continue

                # 读取段长度 (2 bytes)
                if pos + 1 >= len(data):
                    break

                segment_len = struct.unpack(">H", data[pos:pos + 2])[0]
                segment_data = data[pos + 2:pos + segment_len] if segment_len > 2 else b""
                pos += segment_len

                # 处理特定的 APP 段
                if 0xE0 <= marker <= 0xEF:
                    app_name = self._get_app_name(marker)
                    struct.app_segments.append(app_name)

                    if marker == 0xE0 and segment_data.startswith(b"JFIF"):
                        struct.has_jfif = True
                    elif marker == 0xE1 and segment_data.startswith(b"Exif"):
                        struct.has_exif = True
                        # 尝试提取缩略图信息
                        thumb_info = self._extract_thumbnail_info(segment_data)
                        if thumb_info:
                            struct.has_thumbnail = True
                            struct.thumbnail_width = thumb_info.get("width")
                            struct.thumbnail_height = thumb_info.get("height")
                    elif marker == 0xED and segment_data.startswith(b"Photoshop"):
                        struct.has_photoshop = True

                # 处理 DQT (Define Quantization Table) - 量化表指纹
                elif marker == 0xDB:
                    dqt_fingerprint = self._extract_dqt_fingerprint(segment_data)
                    if dqt_fingerprint:
                        struct.dqt_tables.append(dqt_fingerprint)

                # 处理 DHT (Define Huffman Table) - 霍夫曼表指纹
                elif marker == 0xC4:
                    dht_fingerprint = self._extract_dht_fingerprint(segment_data)
                    if dht_fingerprint:
                        struct.dht_tables.append(dht_fingerprint)

                # 处理 SOF (Start of Frame) - 提取尺寸
                elif 0xC0 <= marker <= 0xCF and marker not in [0xC4, 0xC8, 0xCC]:
                    # SOF0, SOF1, SOF2 等
                    if len(segment_data) >= 7:
                        struct.height = struct.unpack(">H", segment_data[1:3])[0]
                        struct.width = struct.unpack(">H", segment_data[3:5])[0]

                # 提取 DQT 估算质量
                if struct.dqt_tables:
                    struct.estimated_quality = self._estimate_quality_from_dqt(struct.dqt_tables[0])

        except Exception as e:
            logger.exception(f"JPEG parse error: {e}")
            struct.structural_errors.append(f"Parse error: {str(e)}")

        return struct

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
        struct = PNGStructure()

        try:
            with open(file_path, "rb") as f:
                data = f.read()

            # PNG 签名检查: 89 50 4E 47 0D 0A 1A 0A
            if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
                struct.structural_errors.append("Invalid PNG signature")
                return struct

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

                struct.chunk_count += 1

                # 关键块
                if chunk_type == "IHDR":
                    struct.critical_chunks.append("IHDR")
                    if len(chunk_data) >= 13:
                        struct.ihdr = {
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
                    struct.has_plte = True
                    struct.critical_chunks.append("PLTE")
                    critical_count += 1

                elif chunk_type == "IDAT":
                    struct.has_idat = True
                    struct.critical_chunks.append("IDAT")
                    critical_count += 1

                elif chunk_type == "IEND":
                    struct.critical_chunks.append("IEND")
                    critical_count += 1

                elif chunk_type == "pHYs":
                    if len(chunk_data) >= 9:
                        struct.phys = {
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
                            struct.text_chunks.append({
                                "type": chunk_type,
                                "keyword": keyword,
                                "value": value[:200]  # 截断长文本
                            })
                    except Exception:
                        pass

            # 检查关键块完整性
            required = ["IHDR", "IDAT", "IEND"]
            missing = [r for r in required if r not in struct.critical_chunks]
            if missing:
                struct.structural_errors.append(f"Missing critical chunks: {missing}")

        except Exception as e:
            logger.exception(f"PNG parse error: {e}")
            struct.structural_errors.append(f"Parse error: {str(e)}")

        return struct