from pydantic import BaseModel
from typing import Optional


class BBox(BaseModel):
    """轴对齐边界框 (x0, y0, x1, y1)，原点为页面左上角"""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def intersects(self, other: "BBox") -> bool:
        """检测两个 bbox 是否相交"""
        return not (self.x1 <= other.x0 or self.x0 >= other.x1 or
                    self.y1 <= other.y0 or self.y0 >= other.y1)

    def contains(self, other: "BBox") -> bool:
        """检测当前 bbox 是否完全包含另一个 bbox"""
        return (self.x0 <= other.x0 and self.x1 >= other.x1 and
                self.y0 <= other.y0 and self.y1 >= other.y1)

    def overlap_ratio(self, other: "BBox") -> float:
        """计算与另一个 bbox 的交集面积占自身面积的比例"""
        if not self.intersects(other):
            return 0.0
        inter_x0 = max(self.x0, other.x0)
        inter_y0 = max(self.y0, other.y0)
        inter_x1 = min(self.x1, other.x1)
        inter_y1 = min(self.y1, other.y1)
        inter_area = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)
        return inter_area / self.area if self.area > 0 else 0.0

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @classmethod
    def from_tuple(cls, tup: tuple) -> "BBox":
        return cls(x0=tup[0], y0=tup[1], x1=tup[2], y1=tup[3])