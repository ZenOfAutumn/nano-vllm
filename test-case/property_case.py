"""
@property 装饰器的详细讲解和正确用法

@property 用于将方法转换为属性，使得调用方式从 obj.method() 变为 obj.method
"""

# ============================================================================
# 1. 错误用法 vs 正确用法
# ============================================================================

# ❌ 错误：@property 不能用在 __init__ 上
# @property 是用来将方法转换为属性访问的
# class Person:
#     @property
#     def __init__(self, name):  # ❌ 错误！这样会破坏初始化
#         self.name = name

# ✅ 正确：使用普通的 __init__ 方法
class Person:
    """正确的 Person 类定义"""

    def __init__(self, name, age):
        """初始化方法"""
        self._name = name  # 🔵 用 _name 表示私有属性
        self._age = age    # 🔵 用下划线前缀表示内部属性


# ============================================================================
# 2. 最简单的 @property 用法
# ============================================================================

class SimpleExample:
    """演示最基础的 @property 用法"""

    def __init__(self, value):
        self._value = value  # 🔵 私有属性（用下划线前缀）

    @property
    def value(self):
        """
        将 value() 方法转换为属性访问。

        用法变化：
        - 旧方式：obj.get_value()
        - 新方式：obj.value（像访问属性一样）
        """
        print("🔍 正在读取 value...")
        return self._value

    @value.setter
    def value(self, new_value):
        """
        允许通过 obj.value = xxx 的方式设置值。

        setter 会在赋值时被自动调用。
        """
        print(f"📝 正在设置 value = {new_value}")
        self._value = new_value

# 使用示例
print("=" * 70)
print("2. 最简单的 @property 用法")
print("=" * 70)
obj = SimpleExample(10)
print(f"读取: {obj.value}")      # 自动调用 getter，输出：正在读取 value...
obj.value = 20                   # 自动调用 setter，输出：正在设置 value = 20
print(f"读取: {obj.value}")      # 输出：正在读取 value...


# ============================================================================
# 3. 实际应用：验证属性赋值
# ============================================================================

class Student:
    """学生类，演示用 @property 验证属性"""

    def __init__(self, name, age):
        self.name = name
        self._age = age  # 🔵 私有属性

    @property
    def age(self):
        """获取年龄"""
        return self._age

    @age.setter
    def age(self, value):
        """
        设置年龄时进行验证。

        这是 @property 的强大用途：在赋值时做验证，
        而不需要调用 set_age(value) 之类的方法。
        """
        if not isinstance(value, int):
            raise TypeError(f"年龄必须是整数，不能是 {type(value)}")
        if value < 0 or value > 150:
            raise ValueError(f"年龄必须在 0-150 之间，不能是 {value}")
        self._age = value

# 使用示例
print("\n" + "=" * 70)
print("3. 验证属性赋值")
print("=" * 70)
s = Student("Alice", 20)
print(f"年龄: {s.age}")  # 20（使用 getter）

s.age = 25              # ✅ 正确，使用 setter
print(f"年龄: {s.age}")  # 25

try:
    s.age = "invalid"   # ❌ 错误，会触发 setter 的验证
except TypeError as e:
    print(f"❌ 错误: {e}")

try:
    s.age = 200         # ❌ 错误，会触发 setter 的验证
except ValueError as e:
    print(f"❌ 错误: {e}")


# ============================================================================
# 4. 计算属性：根据其他属性动态计算
# ============================================================================

class Rectangle:
    """矩形类，演示计算属性"""

    def __init__(self, width, height):
        self._width = width
        self._height = height

    @property
    def width(self):
        """宽度"""
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        """高度"""
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def area(self):
        """
        面积属性：根据宽和高动态计算。

        注意：area 只有 getter，没有 setter（只读属性）
        """
        print("📐 正在计算面积...")
        return self._width * self._height

    @property
    def perimeter(self):
        """周长属性：也是根据宽高动态计算"""
        print("📏 正在计算周长...")
        return 2 * (self._width + self._height)

# 使用示例
print("\n" + "=" * 70)
print("4. 计算属性（只读属性）")
print("=" * 70)
rect = Rectangle(5, 3)
print(f"宽: {rect.width}, 高: {rect.height}")
print(f"面积: {rect.area}")        # 只读，自动计算
print(f"周长: {rect.perimeter}")   # 只读，自动计算

rect.width = 10
print(f"修改宽后的面积: {rect.area}")  # 面积会自动重新计算


# ============================================================================
# 5. 修复原始代码：正确的继承用法
# ============================================================================

class PersonFixed:
    """修复后的 Person 类"""

    def __init__(self, name):
        """初始化方法，不能用 @property"""
        self.__name = name

    @property
    def name(self):
        return self.__name

class Man(PersonFixed):
    """继承 PersonFixed 的 Man 类"""

    def __init__(self, name):
        super().__init__(name)  # 调用父类的 __init__
        self.age = 20


print("\n" + "=" * 70)
print("5. 修复原始代码")
print("=" * 70)
m = Man("John")
print(f"姓名: {m.name}, 年龄: {m.age}")  # 姓名: John, 年龄: 20


# ============================================================================
# 6. @property 的核心概念：用属性访问替代方法调用
# ============================================================================

class BankAccount:
    """银行账户，演示 @property 的完整用途"""

    def __init__(self, balance):
        self._balance = balance  # 🔵 私有属性

    @property
    def balance(self):
        """
        获取余额（只读属性）。

        使用 @property 的好处：
        1. 可以像访问属性一样访问（account.balance）
        2. 但实际上可以在里面做复杂的业务逻辑
        3. 如果以后需要改变实现，外部代码不需要改动
        """
        return self._balance

    def deposit(self, amount):
        """存钱"""
        if amount <= 0:
            raise ValueError("存款金额必须大于 0")
        self._balance += amount
        print(f"✅ 存入 {amount}，余额: {self._balance}")

    def withdraw(self, amount):
        """取钱"""
        if amount <= 0:
            raise ValueError("取款金额必须大于 0")
        if amount > self._balance:
            raise ValueError(f"余额不足（余额: {self._balance}）")
        self._balance -= amount
        print(f"✅ 取出 {amount}，余额: {self._balance}")

# 使用示例
print("\n" + "=" * 70)
print("6. @property 在实际应用中的用途")
print("=" * 70)
account = BankAccount(1000)
print(f"初始余额: {account.balance}")  # 像属性一样访问（不是 balance()）

account.deposit(500)
print(f"存钱后: {account.balance}")

account.withdraw(200)
print(f"取钱后: {account.balance}")


# ============================================================================
# 7. 总结对比
# ============================================================================

print("\n" + "=" * 70)
print("7. @property 总结")
print("=" * 70)

summary = """
❌ 错误用法：
  @property
  def __init__(self):  # 错误！@property 不能用在 __init__
      pass

✅ 正确用法：
  def __init__(self):  # __init__ 必须是普通方法
      pass

  @property
  def some_attr(self):  # 将方法转换为属性访问
      return self._some_attr

  @some_attr.setter
  def some_attr(self, value):  # 允许赋值
      self._some_attr = value

📌 @property 的三个组成部分：
  1. @property - getter，允许读取属性
  2. @attr.setter - setter，允许设置属性
  3. @attr.deleter - deleter，允许删除属性（可选）

💡 何时使用 @property：
  ✅ 需要在读取/设置时做额外处理（验证、计算等）
  ✅ 需要将私有属性公开访问
  ✅ 需要计算属性（如面积、周长）
  ✅ 需要改变实现而不影响外部 API

  ❌ 只是简单地存取数据 → 直接用公开属性
"""

print(summary)
