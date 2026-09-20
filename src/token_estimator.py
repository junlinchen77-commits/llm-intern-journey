"""Token 数量估算模块。

设计取舍：精确计数需要 tokenizer（如 tiktoken）及其模型文件，
本模块用字符级启发式估算，零依赖、可离线运行。
误差范围约 ±30%，用于成本预估与上下文长度初筛。
"""

# Unicode 码点范围，判断字符是否属于中日韩文字
_CJK_RANGES = (
    (0x4E00, 0x9FFF),    # CJK 统一汉字
    (0x3400, 0x4DBF),    # 扩展 A 区
    (0x3040, 0x30FF),    # 日文假名
    (0xAC00, 0xD7AF),    # 韩文音节
)


def _is_cjk(char: str) -> bool:
    """判断单个字符是否为中日韩文字。"""
    code_point = ord(char)
    return any(low <= code_point <= high for low, high in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """估算文本的 token 数量。

    规则（按经验值）：
      - 空文本返回 0
      - 中日韩文字：每字约 1 token
      - 其他字符：每 4 个约 1 token
      - 非空文本至少返回 1

    Args:
        text: 待估算的文本。

    Returns:
        估算出的 token 数量，非负整数。
    """

    if not text:
        return 0

    cjk_count = 0
    other_count = 0

    for char in text:
        if _is_cjk(char):
            cjk_count += 1
        else:
            other_count += 1        

    other_tokens = other_count // 4

    return max(1, cjk_count + other_tokens)

if __name__ == "__main__":
    # 手工验证用，临时替代 pytest
    print(f"空文本: {estimate_tokens('')}")
    print(f"hello: {estimate_tokens('hello')}")
    print(f"你好世界: {estimate_tokens('你好世界')}")