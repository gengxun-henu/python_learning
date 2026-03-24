# Python 源码专项规则

适用于仓库中的 Python 源码文件，例如：

- `examples/**/*.py`
- 根目录下的 `.py` 文件
- 其他非测试 Python 文件

## 总体风格

- 优先写清晰、教学友好的 Python 代码。
- 保留类型注解。
- 优先使用简洁、直观、可读的实现方式。
- 对学习型示例，适度保留解释性输出。
- 优先让函数职责单一，必要时把参数校验逻辑抽成小函数，便于测试。

## Docstring 规则

- Python 函数、方法、类、模块默认使用 **NumPy 风格 docstring**。
- 当用户在 Python 定义下输入三引号 `"""` 时，优先补全完整 docstring。
- 对函数默认包含：
  - 一句中文摘要
  - 空行
  - `Parameters`
  - `Returns`
  - 必要时 `Raises`
- 不要生成 `docstring:`、`Summary:`、`Description:` 这类占位或冗余标签。
- 参数名必须与函数签名保持一致。
- 即使函数已有类型注解，docstring 中仍保留参数和返回类型说明，以保持风格统一。
- 私有小函数可简短，但在参数或返回值不直观时，仍优先补全 `Parameters` 和 `Returns`。

## 推荐格式示例

```python
def ratio_threshold_type(value: str) -> float:
    """解析并校验 Lowe ratio test 阈值。

    Parameters
    ----------
    value : str
        命令行传入的阈值字符串。

    Returns
    -------
    float
        转换并校验后的阈值，要求位于 (0, 1) 区间内。

    Raises
    ------
    argparse.ArgumentTypeError
        当输入不是合法浮点数或超出允许范围时抛出。
    """
```

## 类与模块

- 类 docstring 默认包含：一句中文摘要，以及在确有必要时补充 `Attributes`。
- 模块 docstring 默认包含：文件用途、主要内容、必要的运行方式说明。
- 对教学示例模块，优先解释“做什么”和“如何运行”。

## 风格边界

- 避免为了“看起来高级”而过度抽象。
- 对学习型代码，优先选择容易理解的写法，而不是最炫技的写法。
- 只有在确实有价值时才引入额外复杂结构。
