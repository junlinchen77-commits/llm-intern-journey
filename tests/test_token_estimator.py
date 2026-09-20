import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.token_estimator import estimate_tokens

def test_empty_string_returns_zero():
    assert estimate_tokens("")==0

def test_single_english_word():
    assert estimate_tokens("hello")==1

def test_chinese_text():
    assert estimate_tokens("你好世界")==4

def test_short_english_text_counts_as_one():
    assert estimate_tokens("hi") == 1

if __name__=="__main__":
    test_empty_string_returns_zero()
    test_single_english_word()
    test_chinese_text()
    test_short_english_text_counts_as_one()
    print("all tests passed.")