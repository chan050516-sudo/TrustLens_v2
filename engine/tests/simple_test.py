import sys
from pathlib import Path
from pathlib import Path

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.forensics.reconciliation.rules import get_rules, universal
from app.core.dto_ir import DocumentType
for dt in DocumentType:
    n = len(get_rules(dt)) + 5
    print(f'{dt.value:15s}: {n}')