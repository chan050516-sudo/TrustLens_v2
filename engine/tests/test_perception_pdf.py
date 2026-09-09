import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent  # tests/ 的父目录是 engine/
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.document_ir import DocumentContext
from app.perception import PdfObservationExtractor, ObservationIR

def test_pdf_extraction(file_path: str):
    context = DocumentContext(file_path=Path(file_path))
    extractor = PdfObservationExtractor()
    obs_list = extractor.extract(context)
    
    print(f"Total observations: {len(obs_list)}")
    for i, obs in enumerate(obs_list[:10]):  # 打印前10行
        print(f"[{i+1}] Page {obs.page} | bbox: {obs.bbox.to_tuple()} | text: {obs.text[:50]}...")
    
    # 统计各页数量
    from collections import Counter
    page_counts = Counter(o.page for o in obs_list)
    print(f"Page distribution: {page_counts}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_perception_pdf.py <path_to_pdf>")
        sys.exit(1)
    test_pdf_extraction(sys.argv[1])