"""
输入一个整数 n
输出 
1
1 到 
n
n 的和
判断 n 是奇数还是偶数
输出 1 到 n 的平方表
"""

def main():
    n = int(input("请输入一个整数 n: "))
    
    # 计算 1 到 n 的和
    total_sum = add_numbers(n)
    print(f"1 到 {n} 的和是: {total_sum}")
    
    # 判断 n 是奇数还是偶数
    if is_even(n):
        print(f"{n} 是偶数")
    else:
        print(f"{n} 是奇数")
    
    # 输出 1 到 n 的平方表
    print(f"1 到 {n} 的平方表:")
    for i in range(1, n + 1):
        print(f"{i}^2 = {i**2}")

def add_numbers(n):
    return sum(range(1, n + 1))

def is_even(n):
    return n % 2 == 0

if __name__ == "__main__":
    main()