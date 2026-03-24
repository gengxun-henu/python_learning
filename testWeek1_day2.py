#统计字符串中每个字符出现的次数

def count_chars():
    input_string = input("请输入一个字符串: ")
    char_count = {}
    
    for char in input_string:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
            
    print("字符出现的次数:")
    for char, count in char_count.items():
        print(f"{char}: {count}")

def remove_redundancy(nums):
    #nums= [1,2,1,3,4]
    seen = set()
    unique_nums = []
    for num in nums:
        if num not in seen:
            seen.add(num)
            unique_nums.append(num)
    return unique_nums

#3. 用字典保存学生成绩，并找出最高分学生
def find_top_student(lst):
    students = {}
    for student in lst:
        name = student['name']
        score = student['score']    
        students[name] = score
    
    top_student = max(students, key=students.get)
    return top_student, students[top_student]

"""
能用推导式的地方改成推导式
能直接遍历的地方不用手写下标循环
所有输出尽量改成 f-string
"""

def find_top_student_comprehension(lst):
    """使用字典推导式来创建学生成绩字典，并使用 max 函数找出最高分学生。
    """
    students = {student['name']: student['score'] for student in lst}
    top_student = max(students, key=students.get)
    return top_student, students[top_student]



if __name__ == "__main__":
    #count_chars()
    nums= [1,2,1,3,4]
    uni_nums=remove_redundancy(nums)
    for num in uni_nums:
        print(f"Item  {num}:")

    student_scores1={'name': 'Peter', 'score': 80}
    student_scores2={'name': 'Bob', 'score': 90}
    student_scores3={'name': 'Mary', 'score': 55}

    student_scores=[student_scores1, student_scores2, student_scores3]

    #top_student, top_score = find_top_student(student_scores)
    top_student, top_score = find_top_student_comprehension(student_scores)
    print(f"最高分学生: {top_student}, 分数: {top_score}")