# C++ → Python 代码示例对照版

> 适用对象：有多年 C++ 经验、正在迁移学习 Python 的开发者。  
> 阅读方式：左边看 C++ 习惯，右边看 Python 写法，再看“迁移提示”。

---

## 1. Hello World 与脚本入口

### C++

```cpp
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```

### Python

```python
def main() -> None:
    print("Hello, World!")


if __name__ == "__main__":
    main()
```

### 迁移提示

- Python 不需要编译入口函数，但推荐保留 `main()` 结构
- `if __name__ == "__main__":` 是最常见脚本入口写法
- `print()` 对应最简单的控制台输出

---

## 2. 变量与类型

### C++

```cpp
int age = 42;
double pi = 3.14159;
bool ok = true;
std::string name = "Geng Xun";
```

### Python

```python
age = 42
pi = 3.14159
ok = True
name = "Geng Xun"
```

### 带类型注解的 Python

```python
age: int = 42
pi: float = 3.14159
ok: bool = True
name: str = "Geng Xun"
```

### 迁移提示

- Python 运行时不强制显式声明类型
- 工程代码建议逐步加入类型注解
- `True` / `False` 首字母大写

---

## 3. 条件判断

### C++

```cpp
if (score >= 90) {
    std::cout << "A" << std::endl;
} else if (score >= 60) {
    std::cout << "B" << std::endl;
} else {
    std::cout << "C" << std::endl;
}
```

### Python

```python
if score >= 90:
    print("A")
elif score >= 60:
    print("B")
else:
    print("C")
```

### 迁移提示

- Python 用缩进代替花括号
- 没有圆括号要求，但复杂条件里加括号也可以提高可读性

---

## 4. for 循环：范围遍历

### C++

```cpp
for (int i = 0; i < 5; ++i) {
    std::cout << i << std::endl;
}
```

### Python

```python
for i in range(5):
    print(i)
```

### 迁移提示

- `range(5)` 生成 $0$ 到 $4$
- Python 常见写法比 C++ 更短，但表达的是同一类意图

---

## 5. 遍历容器

### C++

```cpp
std::vector<int> nums{10, 20, 30};
for (const auto& x : nums) {
    std::cout << x << std::endl;
}
```

### Python

```python
nums = [10, 20, 30]
for x in nums:
    print(x)
```

### 同时取索引和值

#### C++

```cpp
for (size_t i = 0; i < nums.size(); ++i) {
    std::cout << i << ": " << nums[i] << std::endl;
}
```

#### Python

```python
for i, x in enumerate(nums):
    print(f"{i}: {x}")
```

### 迁移提示

- Python 更鼓励直接遍历元素
- 需要索引时优先用 `enumerate()`，不要手搓下标循环

---

## 6. 列表与向量

### C++

```cpp
std::vector<int> nums;
nums.push_back(1);
nums.push_back(2);
nums.push_back(3);
```

### Python

```python
nums = []
nums.append(1)
nums.append(2)
nums.append(3)
```

### 过滤偶数

#### C++

```cpp
std::vector<int> even;
for (int x : nums) {
    if (x % 2 == 0) {
        even.push_back(x);
    }
}
```

#### Python

```python
even = [x for x in nums if x % 2 == 0]
```

### 迁移提示

- `list` 对应 `std::vector` 的使用感觉最接近
- 推导式是 Python 的高频写法，值得优先掌握

---

## 7. map / unordered_map 与 dict

### C++

```cpp
std::unordered_map<std::string, int> scores;
scores["Alice"] = 95;
scores["Bob"] = 88;
```

### Python

```python
scores = {"Alice": 95, "Bob": 88}
```

### 遍历键值对

#### C++

```cpp
for (const auto& [name, score] : scores) {
    std::cout << name << ": " << score << std::endl;
}
```

#### Python

```python
for name, score in scores.items():
    print(f"{name}: {score}")
```

### 安全读取默认值

```python
math_score = scores.get("Math", 0)
```

### 迁移提示

- `dict` 是 Python 最核心的数据结构之一
- `items()` 对应键值对遍历
- `get()` 很适合避免 KeyError

---

## 8. set 去重

### C++

```cpp
std::unordered_set<int> s;
s.insert(1);
s.insert(2);
s.insert(2);
```

### Python

```python
s = {1, 2, 2}
print(s)  # {1, 2}
```

### 列表去重

```python
nums = [1, 2, 2, 3, 3, 3]
unique_nums = list(set(nums))
```

### 迁移提示

- `set` 非常适合成员判断和去重
- 但 `set` 本身无序，不要误以为它会保持原始顺序

---

## 9. 字符串格式化

### C++

```cpp
std::string name = "Alice";
int age = 18;
std::cout << name << " is " << age << " years old" << std::endl;
```

### Python

```python
name = "Alice"
age = 18
print(f"{name} is {age} years old")
```

### 迁移提示

- `f-string` 是最推荐的格式化方式
- 比字符串拼接更清晰，也通常更自然

---

## 10. 函数定义与返回值

### C++

```cpp
int add(int a, int b) {
    return a + b;
}
```

### Python

```python
def add(a: int, b: int) -> int:
    return a + b
```

### 多返回值

#### C++

```cpp
std::pair<int, int> minmax(int a, int b) {
    if (a < b) {
        return {a, b};
    }
    return {b, a};
}
```

#### Python

```python
def minmax(a: int, b: int) -> tuple[int, int]:
    if a < b:
        return a, b
    return b, a
```

### 迁移提示

- Python 返回多个值本质上是返回一个 `tuple`
- 解包用起来非常顺手

---

## 11. 默认参数与可变参数

### C++

```cpp
void log(const std::string& msg, int level = 1) {
    std::cout << level << ": " << msg << std::endl;
}
```

### Python

```python
def log(msg: str, level: int = 1) -> None:
    print(f"{level}: {msg}")
```

### Python 可变参数

```python
def add_all(*nums: int) -> int:
    return sum(nums)
```

### 迁移提示

- Python 默认参数非常常见
- `*args` / `**kwargs` 在框架和库中出现频率很高

---

## 12. 一个经典坑：可变默认参数

### 错误写法

```python
def append_item(item: int, items: list[int] = []) -> list[int]:
    items.append(item)
    return items
```

### 为什么有坑

```python
print(append_item(1))  # [1]
print(append_item(2))  # [1, 2]
```

### 正确写法

```python
def append_item(item: int, items: list[int] | None = None) -> list[int]:
    if items is None:
        items = []
    items.append(item)
    return items
```

### 迁移提示

- 默认参数在函数定义时创建，不是调用时创建
- 这是 C++ 程序员初学 Python 的高频坑点之一

---

## 13. 文件读写与 RAII 对照

### C++

```cpp
#include <fstream>
#include <string>

std::ifstream fin("data.txt");
std::string line;
while (std::getline(fin, line)) {
    std::cout << line << std::endl;
}
```

### Python

```python
with open("data.txt", "r", encoding="utf-8") as f:
    for line in f:
        print(line.strip())
```

### 迁移提示

- `with` 是 Python 的资源管理核心语法
- 可以把它理解为通用的 RAII 风格用法

---

## 14. 异常处理

### C++

```cpp
try {
    throw std::runtime_error("something went wrong");
} catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
}
```

### Python

```python
try:
    raise RuntimeError("something went wrong")
except RuntimeError as e:
    print(e)
```

### 带清理逻辑

```python
try:
    value = 10 / x
except ZeroDivisionError:
    print("x 不能为 0")
finally:
    print("无论是否出错都会执行")
```

### 迁移提示

- Python 很依赖异常机制
- 优先捕获具体异常，不建议大面积使用裸 `except:`

---

## 15. 类与对象

### C++

```cpp
class Person {
public:
    Person(std::string name, int age) : name_(std::move(name)), age_(age) {}

    void greet() const {
        std::cout << "Hello, I am " << name_ << std::endl;
    }

private:
    std::string name_;
    int age_;
};
```

### Python

```python
class Person:
    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age

    def greet(self) -> None:
        print(f"Hello, I am {self.name}")
```

### 迁移提示

- `self` 需要显式写出
- Python 没有严格意义的 `private`，更多依赖约定
- 很多轻量场景下更推荐 `dataclass`

---

## 16. dataclass：减少样板代码

### Python

```python
from dataclasses import dataclass


@dataclass
class Person:
    name: str
    age: int
```

### 迁移提示

- 类似“轻量数据对象”的场景优先考虑 `@dataclass`
- 能自动帮你生成初始化和友好表示等样板行为

---

## 17. 属性访问与 property

### C++ 风格直觉

```cpp
class Circle {
public:
    double radius() const { return radius_; }
    void set_radius(double value) { radius_ = value; }

private:
    double radius_;
};
```

### Python

```python
class Circle:
    def __init__(self, radius: float) -> None:
        self._radius = radius

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        if value < 0:
            raise ValueError("radius must be non-negative")
        self._radius = value
```

### 迁移提示

- `property` 让你保持“像访问字段一样访问”，同时保留校验逻辑
- 很适合从 getter/setter 思维迁移过来

---

## 18. 继承与 `super()`

### C++

```cpp
class Animal {
public:
    virtual void speak() const {
        std::cout << "..." << std::endl;
    }
};

class Dog : public Animal {
public:
    void speak() const override {
        std::cout << "woof" << std::endl;
    }
};
```

### Python

```python
class Animal:
    def speak(self) -> None:
        print("...")


class Dog(Animal):
    def speak(self) -> None:
        print("woof")
```

### 使用 `super()`

```python
class Student(Person):
    def __init__(self, name: str, age: int, school: str) -> None:
        super().__init__(name, age)
        self.school = school
```

### 迁移提示

- Python 支持继承，但不鼓励过重的类层次设计
- 很多时候组合比继承更清晰

---

## 19. 列表排序

### C++

```cpp
std::vector<int> nums{4, 2, 5, 1};
std::sort(nums.begin(), nums.end());
```

### Python

```python
nums = [4, 2, 5, 1]
nums.sort()
```

### 带 key 排序

```python
students = [
    {"name": "Alice", "score": 95},
    {"name": "Bob", "score": 88},
]

students.sort(key=lambda x: x["score"], reverse=True)
```

### 迁移提示

- `sort()` 原地排序
- `sorted(...)` 返回新列表
- `key=` 是 Python 排序的核心能力之一

---

## 20. 过滤、映射、聚合

### C++ 直觉

```cpp
std::vector<int> nums{1, 2, 3, 4, 5};
int sum = 0;
for (int x : nums) {
    if (x % 2 == 0) {
        sum += x * x;
    }
}
```

### Python

```python
nums = [1, 2, 3, 4, 5]
result = sum(x * x for x in nums if x % 2 == 0)
```

### 迁移提示

- Python 很擅长用生成式直接表达意图
- 不必凡事都拆成多行循环控制逻辑

---

## 21. 路径处理

### C++ 风格直觉

你可能会手工拼路径字符串。

### Python 推荐

```python
from pathlib import Path

base = Path("data")
file_path = base / "input.txt"

if file_path.exists():
    print(file_path.read_text(encoding="utf-8"))
```

### 迁移提示

- `pathlib` 是现代 Python 路径处理首选
- 尽量少手工写字符串拼路径

---

## 22. 泛型/模板思维迁移

### C++

```cpp
template <typename T>
T maximum(T a, T b) {
    return a > b ? a : b;
}
```

### Python：最直接的做法

```python
def maximum(a, b):
    return a if a > b else b
```

### Python：带类型提示

```python
from typing import TypeVar

T = TypeVar("T")


def maximum(a: T, b: T) -> T:
    return a if a > b else b
```

### 迁移提示

- Python 常常先写“能工作”的泛化逻辑
- 再按需要加类型提示，而不是一开始就做复杂模板设计

---

## 23. 接口约束：从 Concepts 思维到 Protocol

### Python

```python
from typing import Protocol


class SupportsClose(Protocol):
    def close(self) -> None:
        ...


def shutdown(resource: SupportsClose) -> None:
    resource.close()
```

### 迁移提示

- `Protocol` 很适合表达“只要你有这个方法，就能被使用”
- 这是从 C++ 类型约束思维迁移到 Python 结构化接口思维的关键一步

---

## 24. 一个综合小例子：统计文本单词频率

### Python

```python
from collections import Counter
from pathlib import Path


def top_words(file_path: str, top_n: int = 10) -> list[tuple[str, int]]:
    text = Path(file_path).read_text(encoding="utf-8")
    words = text.lower().split()
    counter = Counter(words)
    return counter.most_common(top_n)


if __name__ == "__main__":
    for word, count in top_words("data.txt"):
        print(f"{word}: {count}")
```

### 这个例子体现了什么

- 函数定义
- 默认参数
- 类型注解
- `pathlib`
- 标准库 `Counter`
- tuple 解包
- `f-string`

这类代码很能代表 Python 的风格：短、直接、表达力强。

---

## 最后给你的迁移建议

如果你每看一个例子都顺手问自己这三个问题，进步会很快：

1. 这个例子在 C++ 里通常怎么写？
2. Python 为什么可以写得更短？
3. 哪种写法在 Python 里更自然、更可维护？

---

## 一页版结论

- Python 更强调表达意图，而不是样板结构
- `list / dict / set` 是高频主力
- `with` 可以视作 Python 的通用 RAII 入口
- `dataclass` 非常适合轻量类
- `enumerate`、`zip`、推导式值得尽快练熟
- 默认参数、可变对象、共享引用是新手高频坑点
- 先写清楚，再谈优化
