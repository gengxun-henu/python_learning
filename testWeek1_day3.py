"""
读取一个文本文件
统计行数、单词数、字符数
输出前 10 个高频词
"""

from collections import Counter


def count_file(filename):
    with open(filename, 'r') as file:
        text = file.read()
        
    lines = text.splitlines()
    words = text.split()
    characters = len(text)
    
    print(f"行数: {len(lines)}")
    print(f"单词数: {len(words)}")
    print(f"字符数: {characters}")
    
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(10)
    
    print("前 10 个高频词:")
    for word, count in most_common_words:
        print(f"{word}: {count}")


"""
能用推导式的地方改成推导式
能直接遍历的地方不用手写下标循环
所有输出尽量改成 f-string
"""

# 用推导式统计单词频率
# 下面是 count_file_comprehension 函数的实现
def count_file_comprehension(filename):
    with open(filename, 'r') as file:
        text = file.read()
        
    lines = text.splitlines()
    words = text.split()
    characters = len(text)
    
    print(f"行数: {len(lines)}")
    print(f"单词数: {len(words)}")
    print(f"字符数: {characters}")
    
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(10)
    
    print("前 10 个高频词:")
    for word, count in most_common_words:
        print(f"{word}: {count}")   

    
if __name__ == "__main__":
    filename = input("请输入文件名: ")
    #count_file(filename)
    count_file_comprehension(filename)