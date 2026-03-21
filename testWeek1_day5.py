
"""
实现一个简单几何模型：

- `Point`
- `Rectangle`
- 支持面积计算
- 打印友好字符串
- 判断点是否落在矩形内部

- `__repr__`
- `@dataclass`
"""

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Point({self.x}, {self.y})"

# 下面是 Rectangle 类的实现
# 注意：这里没有使用 @dataclass，因为我们需要自定义方法和字符串表示
class Rectangle:
    def __init__(self, bottom_left, top_right):
        self.bottom_left = bottom_left
        self.top_right = top_right

    def area(self):
        width = self.top_right.x - self.bottom_left.x
        height = self.top_right.y - self.bottom_left.y
        return width * height

    def contains(self, point):
        return (self.bottom_left.x <= point.x <= self.top_right.x and
                self.bottom_left.y <= point.y <= self.top_right.y)

    def __str__(self):
        return f"Rectangle({self.bottom_left}, {self.top_right})"

if __name__ == "__main__":
    rect = Rectangle(Point(0, 0), Point(4, 3))
    print(rect)
    print(f"Area: {rect.area()}")

    p1 = Point(2, 2)
    p2 = Point(5, 5)
    print(f"{p1} is inside the rectangle: {rect.contains(p1)}")
    print(f"{p2} is inside the rectangle: {rect.contains(p2)}")