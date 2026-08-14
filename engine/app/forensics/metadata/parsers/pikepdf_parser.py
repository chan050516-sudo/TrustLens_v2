# engine/app/forensics/metadata/parsers/pikepdf_parser.py
import logging
from typing import Dict, Any, List, Tuple

import pikepdf
from pikepdf import Dictionary, Name, Stream, Reference

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
        try:
            # 打开 PDF，只读模式
            with pikepdf.open(file_path, allow_overwriting_input=False) as pdf:
                # 统计对象总数
                total_objs = len(pdf.objects)
                graph.total_objects = total_objs

                # 遍历所有对象，检测流和特殊对象
                embedded_files = []
                js_refs = []
                launch_refs = []
                open_action_ref = None

                for ref in pdf.objects:
                    obj = pdf.objects[ref]
                    obj_id = ref.objgen[0]

                    if isinstance(obj, Stream):
                        graph.total_streams += 1

                        # 检测嵌入文件：类型为 /Filespec 且含 /EF，或直接包含 /EmbeddedFile
                        is_embedded = False
                        file_name = None

                        # 检查 /Type
                        if "/Type" in obj and obj["/Type"] == "/Filespec":
                            # 有 /EF 键表示嵌入文件
                            if "/EF" in obj:
                                is_embedded = True
                                # 获取文件名 (可能为 /UF 或 /F)
                                file_name = obj.get("/UF", None)
                                if file_name is None:
                                    file_name = obj.get("/F", None)
                                if isinstance(file_name, bytes):
                                    file_name = file_name.decode(errors="ignore")

                        # 也检查直接包含 /EmbeddedFile (较少见)
                        if "/EmbeddedFile" in obj:
                            is_embedded = True
                            file_name = obj.get("/F", None)
                            if isinstance(file_name, bytes):
                                file_name = file_name.decode(errors="ignore")

                        if is_embedded:
                            embedded_files.append({
                                "id": str(obj_id),
                                "name": file_name or f"embedded_{obj_id}"
                            })

                        # 检测 JavaScript
                        if "/JavaScript" in obj or "/JS" in obj:
                            js_refs.append(str(obj_id))

                        # 检测 Launch
                        if "/Launch" in obj:
                            launch_refs.append(str(obj_id))

                # 检查根部的 /OpenAction
                if "/OpenAction" in pdf.Root:
                    open_action_ref = str(pdf.Root["/OpenAction"])

                graph.embedded_files = embedded_files
                graph.javascript_present = len(js_refs) > 0
                graph.launch_actions_present = len(launch_refs) > 0
                graph.open_action_present = open_action_ref is not None

                # 构建页面 -> XObject 边
                try:
                    for page_num, page in enumerate(pdf.pages, start=1):
                        # 获取页面对象ID
                        page_ref = page.objgen[0]  # 实际上 page 是 pikepdf.IndirectObject，其 objgen 是 (id, gen)
                        # 但我们可以获取其引用
                        # 更好：使用 page._ref 或 page.objgen
                        if hasattr(page, 'objgen'):
                            page_id = page.objgen[0]
                        else:
                            # fallback: 通过查找
                            page_id = None

                        resources = page.get("/Resources", {})
                        xobjects = resources.get("/XObject", {})
                        if isinstance(xobjects, Dictionary):
                            xobject_ids = []
                            for xobj_name, xobj_ref in xobjects.items():
                                if isinstance(xobj_ref, Reference):
                                    xobject_ids.append(xobj_ref.objgen[0])
                            if xobject_ids:
                                graph.pages_with_xobjects[page_num] = xobject_ids
                                if page_id is not None:
                                    for xid in xobject_ids:
                                        graph.edges.append((page_id, xid))
                except Exception as e:
                    logger.warning(f"Error extracting page XObjects: {e}")

        except pikepdf.PdfError as e:
            error_msg = f"pikepdf error: {e}"
            logger.error(error_msg)
            graph.error = error_msg
        except Exception as e:
            error_msg = f"Unexpected error in pikepdf parser: {e}"
            logger.exception(error_msg)
            graph.error = error_msg

        # 返回包含 object_graph 的字典，也可附带其他元数据
        return {"object_graph": graph}