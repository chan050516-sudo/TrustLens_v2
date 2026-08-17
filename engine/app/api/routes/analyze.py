# engine/app/api/routes/analyze.py
"""
分析 API 端点
使用 ForensicRunner 处理请求
"""
import json
import logging
from pathlib import Path
import tempfile
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse

from app.orchestration.runner import ForensicRunner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["analysis"])


@router.post("/sync")
async def analyze_sync(
    file: UploadFile = File(...),
    document_type: Optional[str] = Form(None),
    expected_company: Optional[str] = Form(None),
):
    """
    同步分析端点
    上传文件，等待分析完成返回结果
    """
    try:
        # 保存临时文件
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        # 执行分析
        result = await ForensicRunner.analyze(
            file_path=tmp_path,
            document_type=document_type,
            expected_company=expected_company,
        )
        
        # 清理临时文件
        try:
            tmp_path.unlink()
        except Exception:
            pass
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def analyze_stream(
    file: UploadFile = File(...),
    document_type: Optional[str] = Form(None),
    expected_company: Optional[str] = Form(None),
):
    """
    流式分析端点 (SSE)
    实时返回各阶段的进度和证据
    """
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    
    async def event_generator():
        try:
            async for event in ForensicRunner.analyze_stream(
                file_path=tmp_path,
                document_type=document_type,
                expected_company=expected_company,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/health")
async def health():
    return {"status": "ok"}