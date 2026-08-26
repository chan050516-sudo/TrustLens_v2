# engine/app/forensics/metadata/parsers/pikepdf_parser.py
import logging
from typing import Dict, Any

import pikepdf
from pikepdf import Dictionary, Stream, Object, Array

from app.core.document_ir import DocumentContext
from app.forensics.metadata.interfaces import BaseParser
from app.forensics.metadata.models.metadata_ir import ObjectGraph

logger = logging.getLogger(__name__)


class PikepdfParser(BaseParser):
    """使用 pikepdf 构建 PDF 对象图，检测嵌入文件和可疑动作 (D, G)"""

    def name(self) -> str:
        return "pikepdf"

    def parse(self, context: DocumentContext) -> Dict[str, Any]:
        file_path = context.file_path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        graph = ObjectGraph()
        has_acroform = False
        has_layers = False
        has_annotations = False
        object_stream_count = 0

        try:
            # 🔴 核心修复：所有操作必须内聚在 with 上下文管理器内部
            with pikepdf.open(file_path, allow_overwriting_input=False) as pdf:
                # 统计对象总数
                graph.total_objects = len(pdf.objects)

                embedded_files = []
                js_refs = []
                launch_refs = []
                open_action_ref = None

                # ✅ 修复 1：正确遍历 pdf.objects
                # 直接迭代 pdf.objects 获取 Object 对象
                for obj in pdf.objects:
                    if obj is None:
                        continue

                    # 安全获取对象编号（只有间接对象才有 objgen）
                    obj_id = obj.objgen[0] if getattr(obj, "is_indirect", False) else None
                    if obj_id is None:
                        continue

                    # 统计流对象
                    if isinstance(obj, Stream):
                        graph.total_streams += 1

                        # 检测对象流 (/ObjStm)
                        obj_type = str(obj.get("/Type", ""))
                        if obj_type == "/ObjStm" or ("/First" in obj and "/N" in obj):
                            object_stream_count += 1

                    # ✅ 修复 2：同时检测 Dictionary 和 Stream（不再只限制 Stream）
                    if isinstance(obj, (Dictionary, Stream)):
                        obj_type = str(obj.get("/Type", ""))

                        # --- 检测嵌入文件 ---
                        if obj_type == "/Filespec" or "/EF" in obj or "/EmbeddedFile" in obj:
                            file_name = None
                            for key in ("/UF", "/F", "/Unix", "/DOS"):
                                if key in obj:
                                    # 处理可能为 bytes 或 Object 的值
                                    raw_val = obj[key]
                                    if isinstance(raw_val, bytes):
                                        file_name = raw_val.decode("utf-8", errors="ignore")
                                    else:
                                        file_name = str(raw_val)
                                    break

                            embedded_files.append({
                                "id": str(obj_id),
                                "name": file_name or f"embedded_{obj_id}"
                            })

                        # --- 检测 JavaScript ---
                        if "/JavaScript" in obj or "/JS" in obj:
                            js_refs.append(str(obj_id))
                        # 也检查动作类型 (S 键)
                        action_s = str(obj.get("/S", ""))
                        if action_s == "/JavaScript":
                            js_refs.append(str(obj_id))

                        # --- 检测 Launch ---
                        if "/Launch" in obj:
                            launch_refs.append(str(obj_id))
                        if action_s == "/Launch":
                            launch_refs.append(str(obj_id))

                # ✅ 修复 3：全局属性检测（Root 级别）
                if "/OpenAction" in pdf.Root:
                    open_action_ref = str(pdf.Root["/OpenAction"])
                if "/AcroForm" in pdf.Root:
                    has_acroform = True
                if "/OCProperties" in pdf.Root:
                    has_layers = True

                graph.embedded_files = embedded_files
                graph.javascript_present = len(js_refs) > 0
                graph.launch_actions_present = len(launch_refs) > 0
                graph.open_action_present = open_action_ref is not None

                # ✅ 修复 4：页面维度的遍历（XObjects + 注释）
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_id = page.objgen[0] if getattr(page, "is_indirect", False) else None

                    # 注释检测
                    if "/Annots" in page:
                        annots = page.get("/Annots")
                        if isinstance(annots, Array) and len(annots) > 0:
                            has_annotations = True

                    # XObject 解析（构建对象图边）
                    resources = page.get("/Resources")
                    if isinstance(resources, (Dictionary, Object)):
                        xobjects = resources.get("/XObject")
                        if isinstance(xobjects, (Dictionary, Object)):
                            xobject_ids = []
                            # 迭代 xobjects 中的值
                            for _, xobj in xobjects.items():
                                if getattr(xobj, "is_indirect", False):
                                    xobject_ids.append(xobj.objgen[0])

                            if xobject_ids:
                                graph.pages_with_xobjects[page_num] = xobject_ids
                                if page_id is not None:
                                    for xid in xobject_ids:
                                        graph.edges.append((page_id, xid))

        except pikepdf.PdfError as e:
            error_msg = f"pikepdf error: {e}"
            logger.error(error_msg)
            graph.error = error_msg
        except Exception as e:
            error_msg = f"Unexpected error in pikepdf parser: {e}"
            logger.exception(error_msg)
            graph.error = error_msg

        # 所有 result 组装都在 with 块外，但数据已经安全保存到变量中
        return {
            "object_graph": graph,
            "has_acroform": has_acroform,
            "has_layers": has_layers,
            "has_annotations": has_annotations,
            "object_stream_count": object_stream_count,
            "total_objects": graph.total_objects
        }