"""容差常数。

金额 / 百分比 / 数量的比较使用 Decimal 精确运算，
容差用于吸收 VLM 读取时的微小误差。
"""
from decimal import Decimal

MONEY_TOLERANCE = Decimal("0.01")          # 一分钱
PERCENTAGE_TOLERANCE = Decimal("0.01")     # 1 pp
QUANTITY_TOLERANCE = Decimal("0.001")
DATE_TOLERANCE_DAYS = 0                    # 日期不允许误差