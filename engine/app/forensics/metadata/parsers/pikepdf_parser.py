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

                # ===== 新增：注释提取 (指南 §3.10, §4.5) =====
                annotations_pikepdf = []
                for page_num, page in enumerate(pdf.pages, start=1):
                    if "/Annots" in page:
                        annots = page["/Annots"]
                        if isinstance(annots, Array) and len(annots) > 0:
                            for annot_ref in annots:
                                try:
                                    annot_obj = annot_ref  # 直接使用
                                    subtype = str(annot_obj.get("/Subtype", ""))
                                    rect = annot_obj.get("/Rect")
                                    bbox = [rect[0], rect[1], rect[2], rect[3]] if rect else None
                                    
                                    # 提取 URI (如果是 Link)
                                    uri = None
                                    action = None
                                    if subtype == "/Link":
                                        action_dict = annot_obj.get("/A")
                                        if action_dict:
                                            action = dict(action_dict)
                                            if "/URI" in action_dict:
                                                uri = str(action_dict["/URI"])
                                    # 提取文本内容 (如果是 Text)
                                    content = None
                                    if subtype == "/Text":
                                        content = str(annot_obj.get("/Contents", ""))
                                    # 提取表单字段 (如果是 Widget)
                                    field_name = None
                                    if subtype == "/Widget":
                                        field_name = str(annot_obj.get("/T", ""))
                                    
                                    annotations_pikepdf.append({
                                        "page": page_num,
                                        "type": subtype.replace("/", ""),
                                        "uri": uri,
                                        "action": action,
                                        "content": content[:500] if content else None,
                                        "bbox": bbox,
                                        "field_name": field_name,
                                        "source": "pikepdf",
                                    })
                                except Exception as e:
                                    logger.debug(f"Error extracting pikepdf annotation: {e}")

                # ===== 新增：附加动作 (指南 §4.3) =====
                additional_actions = []
                if "/AA" in pdf.Root:
                    aa_dict = pdf.Root["/AA"]
                    if isinstance(aa_dict, Dictionary):
                        for action_key, action_ref in aa_dict.items():
                            try:
                                action_obj = action_ref  # 直接使用
                                action_type = str(action_key)
                                action_s = str(action_obj.get("/S", ""))
                                additional_actions.append({
                                    "trigger": action_type,
                                    "action_type": action_s,
                                    "detail": dict(action_obj) if isinstance(action_obj, Dictionary) else None,
                                })
                            except Exception as e:
                                logger.debug(f"Error extracting additional action: {e}")

                # ===== 新增：嵌入文件提取 (指南 §4.4) =====
                embedded_files_detail = []
                if "/Names" in pdf.Root and "/EmbeddedFiles" in pdf.Root["/Names"]:
                    try:
                        names_dict = pdf.Root["/Names"]["/EmbeddedFiles"]
                        if "/Names" in names_dict:
                            names_list = names_dict["/Names"]
                            for i in range(0, len(names_list), 2):
                                name = str(names_list[i])
                                ref = names_list[i+1]
                                # 解析文件规格
                                if hasattr(ref, "get"):
                                    filespec = ref
                                    ef = filespec.get("/EF", {})
                                    if "/F" in ef:
                                        file_ref = ef["/F"]
                                        # 尝试获取大小
                                        size = len(file_ref.get_bytes()) if hasattr(file_ref, "get_bytes") else None
                                        embedded_files_detail.append({
                                            "name": name,
                                            "size": size,
                                            "mime": str(filespec.get("/MIMEType", "unknown")),
                                            "xref": str(file_ref.objgen[0]) if hasattr(file_ref, "objgen") else None,
                                        })
                    except Exception as e:
                        logger.warning(f"Error extracting embedded files: {e}")

                # ===== 新增：活跃内容提取 (指南 §4.3) =====
                active_content_detail = {
                    "javascript": False,
                    "open_action": False,
                    "launch_action": False,
                    "script_hash": None,
                    "script_snippet": None,
                }
                # 检查 OpenAction
                if "/OpenAction" in pdf.Root:
                    active_content_detail["open_action"] = True
                # 检查 JavaScript
                js_objects = []
                for obj in pdf.objects:
                    if isinstance(obj, (Dictionary, Stream)):
                        if "/JavaScript" in obj or "/JS" in obj:
                            js_objects.append(obj)
                if js_objects:
                    active_content_detail["javascript"] = True
                    # 尝试提取脚本片段
                    for js_obj in js_objects[:3]:  # 最多处理3个
                        try:
                            if isinstance(js_obj, Stream):
                                script = js_obj.get_bytes().decode("utf-8", errors="ignore")
                                if len(script) < 1000:
                                    active_content_detail["script_snippet"] = script[:500]
                                else:
                                    import hashlib
                                    active_content_detail["script_hash"] = hashlib.sha256(script.encode()).hexdigest()[:16]
                        except Exception:
                            pass
                # 检查 Launch
                for obj in pdf.objects:
                    if isinstance(obj, (Dictionary, Stream)):
                        if "/Launch" in obj:
                            active_content_detail["launch_action"] = True
                            break

                # ===== 修复：使用迭代式栈遍历替代递归 =====
                orphan_objects = []
                # 构建 Root 可达对象集合
                reachable = set()
                stack = [pdf.Root]
                
                while stack:
                    obj = stack.pop()
                    if obj is None:
                        continue
                    obj_id = id(obj)
                    if obj_id in reachable:
                        continue
                    reachable.add(obj_id)
                    
                    # 遍历子对象
                    if isinstance(obj, (Dictionary, Stream)):
                        for val in obj.values():
                            if isinstance(val, (Dictionary, Stream, Array)):
                                stack.append(val)
                            elif hasattr(val, "objgen") and val.is_indirect:
                                stack.append(val)
                    elif isinstance(obj, Array):
                        for item in obj:
                            if isinstance(item, (Dictionary, Stream, Array)):
                                stack.append(item)
                            elif hasattr(item, "objgen") and item.is_indirect:
                                stack.append(item)
                    # 注意：某些对象可能是直接间接引用，需要额外处理
                    elif hasattr(obj, "objgen") and obj.is_indirect:
                        # 如果是间接对象，尝试解析它
                        try:
                            resolved = obj()
                            if resolved is not None:
                                stack.append(resolved)
                        except Exception:
                            pass

                # 然后检查所有对象
                try:
                    for obj in pdf.objects:
                        if id(obj) not in reachable:
                            obj_id = obj.objgen[0] if hasattr(obj, "objgen") else "unknown"
                            snippet = None
                            if isinstance(obj, Stream):
                                try:
                                    snippet = obj.get_bytes().decode("utf-8", errors="ignore")[:200]
                                except Exception:
                                    snippet = "[binary data]"
                            elif isinstance(obj, Dictionary):
                                # ✅ 修复：简化 snippet 提取，只查关键文本字段
                                text_keys = ["/Contents", "/Text", "/Value", "/Desc", "/Name", "/Title", "/Subject"]
                                text_values = []
                                for key in text_keys:
                                    if key in obj:
                                        val = obj[key]
                                        if isinstance(val, str):
                                            text_values.append(f"{key}: {val[:80]}")
                                        elif isinstance(val, bytes):
                                            try:
                                                text_values.append(f"{key}: {val.decode('utf-8', errors='ignore')[:80]}")
                                            except Exception:
                                                pass
                                        # 限制最多提取 3 个键值对
                                        if len(text_values) >= 3:
                                            break
                                if text_values:
                                    snippet = " | ".join(text_values)[:300]
                            orphan_objects.append({
                                "xref": str(obj_id),
                                "type": "stream" if isinstance(obj, Stream) else "dictionary",
                                "size": len(obj.get_bytes()) if isinstance(obj, Stream) else len(str(obj)),
                                "semantic_snippet": snippet,
                            })
                except Exception as e:
                    logger.warning(f"Error during orphan object detection: {e}")

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
            "total_objects": graph.total_objects,
            "embedded_files_detail": embedded_files_detail,
            "active_content_detail": active_content_detail,
            "orphan_objects": orphan_objects,
            "annotations_pikepdf": annotations_pikepdf,
            "additional_actions": additional_actions,
        }