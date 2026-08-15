"""
Layer 1 Metadata Forensics 自定义异常
"""


class MetadataForensicsError(Exception):
    """L1 元数据取证的基类异常"""
    pass


class CollectorError(MetadataForensicsError):
    """收集器（Collector）相关异常"""
    pass


class ParserError(MetadataForensicsError):
    """解析器（Parser）相关异常"""
    pass


class AnalyzerError(MetadataForensicsError):
    """分析器（Analyzer）相关异常"""
    pass


class ToolNotFoundError(CollectorError):
    """系统工具未安装或不在 PATH 中"""
    def __init__(self, tool_name: str, installation_hint: str = ""):
        self.tool_name = tool_name
        self.installation_hint = installation_hint
        super().__init__(
            f"Required tool '{tool_name}' not found in PATH. {installation_hint}"
        )


class ExifToolNotFoundError(ToolNotFoundError):
    def __init__(self):
        super().__init__(
            "exiftool",
            "Install: sudo apt install exiftool (Linux) / brew install exiftool (macOS) / download from exiftool.org (Windows)"
        )


class QPDFNotFoundError(ToolNotFoundError):
    def __init__(self):
        super().__init__(
            "qpdf",
            "Install: sudo apt install qpdf (Linux) / brew install qpdf (macOS) / download from qpdf.sourceforge.io (Windows)"
        )


class PDFSigNotFoundError(ToolNotFoundError):
    def __init__(self):
        super().__init__(
            "pdfsig",
            "Install: sudo apt install poppler-utils (Linux) / brew install poppler (macOS)"
        )


class PDFParseError(ParserError):
    """PDF 解析失败"""
    def __init__(self, message: str, original_error: Exception = None):
        self.original_error = original_error
        super().__init__(f"PDF parse error: {message}")


class XMPParseError(ParserError):
    """XMP 元数据解析失败"""
    def __init__(self, message: str):
        super().__init__(f"XMP parse error: {message}")


class SignatureVerificationError(ParserError):
    """数字签名验证失败"""
    def __init__(self, message: str):
        super().__init__(f"Signature verification error: {message}")


class ConfigurationError(MetadataForensicsError):
    """配置错误（如指纹库加载失败）"""
    pass