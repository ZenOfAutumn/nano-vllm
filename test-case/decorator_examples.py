"""
Python 装饰器完整示例
装饰器是一个函数，它接收另一个函数或类作为参数，并返回一个增强版本的函数或类
"""

import functools
import time


# ============================================================================
# 1. 最简单的装饰器：打印函数执行信息
# ============================================================================

def simple_decorator(func):
    """最简单的装饰器：在函数前后输出信息"""
    # *args - 可变位置参数 **kwargs - 可变关键字参数
    def wrapper(*args, **kwargs):
        print(f"🔥 开始执行函数: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"✅ 函数执行完成: {func.__name__}")
        return result
    return wrapper


@simple_decorator
def greet(name):
    """问候函数"""
    print(f"   Hello, {name}!")
    return f"Hi {name}"


# 使用示例
print("=" * 60)
print("1. 简单装饰器例子")
print("=" * 60)
greet("Alice")
# 输出：
# 🔥 开始执行函数: greet
#    Hello, Alice!
# ✅ 函数执行完成: greet


# ============================================================================
# 2. 测量函数执行时间的装饰器
# ============================================================================

def timer_decorator(func):
    """测量函数执行时间"""
    @functools.wraps(func)  # 保留原函数的元信息
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"⏱️  {func.__name__} 执行耗时: {elapsed:.4f} 秒")
        return result
    return wrapper


@timer_decorator
def slow_function():
    """一个耗时的函数"""
    time.sleep(0.5)
    return "完成"


print("\n" + "=" * 60)
print("2. 计时装饰器例子")
print("=" * 60)
slow_function()
# 输出：⏱️  slow_function 执行耗时: 0.5010 秒


# ============================================================================
# 3. 带参数的装饰器
# ============================================================================

def repeat_decorator(times):
    """重复执行函数 N 次的装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for i in range(times):
                print(f"第 {i+1} 次执行...")
                result = func(*args, **kwargs)
                results.append(result)
            return results
        return wrapper
    return decorator


@repeat_decorator(times=3)
def task():
    """要重复执行的任务"""
    return "任务完成"


print("\n" + "=" * 60)
print("3. 带参数的装饰器例子")
print("=" * 60)
results = task()
print(f"执行结果: {results}")
# 输出：
# 第 1 次执行...
# 第 2 次执行...
# 第 3 次执行...
# 执行结果: ['任务完成', '任务完成', '任务完成']


# ============================================================================
# 4. 参数验证装饰器
# ============================================================================

def validate_types(**type_checks):
    """验证函数参数类型的装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 检查关键字参数
            for param_name, expected_type in type_checks.items():
                if param_name in kwargs:
                    value = kwargs[param_name]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"参数 '{param_name}' 应该是 {expected_type.__name__} 类型，"
                            f"但得到 {type(value).__name__} 类型"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator


@validate_types(age=int, name=str)
def create_user(name, age):
    """创建用户"""
    print(f"✅ 创建用户: {name}, 年龄: {age}")
    return {"name": name, "age": age}


print("\n" + "=" * 60)
print("4. 参数验证装饰器例子")
print("=" * 60)
create_user("张三", age=25)  # ✅ 正确
try:
    create_user("李四", age="invalid")  # ❌ 错误
except TypeError as e:
    print(f"❌ 错误: {e}")


# ============================================================================
# 5. 缓存装饰器（记忆化）
# ============================================================================

def cache_decorator(func):
    """缓存函数结果，相同参数只计算一次"""
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 生成缓存键
        cache_key = (args, tuple(sorted(kwargs.items())))

        if cache_key in cache:
            print(f"💾 从缓存返回结果")
            return cache[cache_key]

        print(f"🔄 计算新结果...")
        result = func(*args, **kwargs)
        cache[cache_key] = result
        return result

    return wrapper


@cache_decorator
def fibonacci(n):
    """计算斐波那契数列"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


print("\n" + "=" * 60)
print("5. 缓存装饰器例子")
print("=" * 60)
print(f"fibonacci(5) = {fibonacci(5)}")
print(f"fibonacci(5) = {fibonacci(5)}")  # 第二次会使用缓存


# ============================================================================
# 6. 异常处理装饰器
# ============================================================================

def error_handler(func):
    """捕获异常并处理"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ZeroDivisionError:
            print(f"❌ 错误: 除数不能为零")
            return None
        except Exception as e:
            print(f"❌ 未知错误: {e}")
            return None
    return wrapper


@error_handler
def divide(a, b):
    """除法"""
    return a / b


print("\n" + "=" * 60)
print("6. 异常处理装饰器例子")
print("=" * 60)
print(f"10 / 2 = {divide(10, 2)}")  # ✅ 正常
print(f"10 / 0 = {divide(10, 0)}")  # ❌ 捕获异常


# ============================================================================
# 7. 类装饰器
# ============================================================================

def add_methods(cls):
    """为类添加方法的装饰器"""
    def to_string(self):
        return f"{cls.__name__}({self.__dict__})"

    def is_equal(self, other):
        return self.__dict__ == other.__dict__

    cls.__str__ = to_string
    cls.__eq__ = is_equal
    return cls


@add_methods
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age


print("\n" + "=" * 60)
print("7. 类装饰器例子")
print("=" * 60)
p1 = Person("Alice", 25)
p2 = Person("Alice", 25)
print(f"p1: {p1}")
print(f"p1 == p2: {p1 == p2}")


# ============================================================================
# 8. 实际应用：日志装饰器（最常用）
# ============================================================================

def log_decorator(func):
    """为函数添加日志记录"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"📝 [日志] 调用函数: {func.__name__}")
        print(f"        参数: args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            print(f"📝 [日志] 返回值: {result}")
            return result
        except Exception as e:
            print(f"📝 [日志] 异常: {type(e).__name__}: {e}")
            raise
    return wrapper


@log_decorator
def process_data(data, multiply=1):
    """处理数据"""
    return data * multiply


print("\n" + "=" * 60)
print("8. 日志装饰器例子（实际应用）")
print("=" * 60)
process_data(10, multiply=2)


# ============================================================================
# 9. 多个装饰器组合使用
# ============================================================================

@timer_decorator
@log_decorator
def complex_operation(x, y):
    """使用多个装饰器"""
    time.sleep(0.1)
    return x + y


print("\n" + "=" * 60)
print("9. 多个装饰器组合例子")
print("=" * 60)
result = complex_operation(5, 3)
# 装饰器执行顺序：从下到上应用
# 1. 先应用 log_decorator
# 2. 再应用 timer_decorator


# ============================================================================
# 10. 使用 functools.wraps 保留函数元信息
# ============================================================================

def bad_decorator(func):
    """❌ 不使用 @functools.wraps"""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def good_decorator(func):
    """✅ 使用 @functools.wraps"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@bad_decorator
def bad_func():
    """原始文档"""
    pass


@good_decorator
def good_func():
    """原始文档"""
    pass


print("\n" + "=" * 60)
print("10. functools.wraps 的重要性")
print("=" * 60)
print(f"bad_func.__name__ = {bad_func.__name__}")    # wrapper（❌ 失去原名）
print(f"bad_func.__doc__ = {bad_func.__doc__}")      # None（❌ 失去文档）
print(f"good_func.__name__ = {good_func.__name__}")  # good_func（✅ 保留原名）
print(f"good_func.__doc__ = {good_func.__doc__}")    # 原始文档（✅ 保留文档）


# ============================================================================
# 总结
# ============================================================================

print("\n" + "=" * 60)
print("装饰器核心要点总结")
print("=" * 60)
print("""
1. 装饰器是函数，接收函数/类作为参数，返回增强版本
2. 语法糖：@decorator 等价于 func = decorator(func)
3. 常用场景：日志、计时、参数验证、缓存、异常处理
4. 带参数装饰器：需要三层函数嵌套
5. 一定要使用 @functools.wraps 保留原函数信息
6. 多个装饰器：执行顺序从下到上应用
7. 可以装饰函数和类

装饰器模式是 Python 中非常重要的高级特性！
""")

