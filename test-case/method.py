class Person:
    def __init__(self, name):
        self.name = name
        self.__age = 0  # 私有属性

    def __str__(self):  # 魔术方法
        return f"Person: {self.name}"



    def __len__(self):  # 魔术方法
        return 100
    def __private_method(self):  # 私有方法
        return f"年龄是私密的"



p = Person("张三")
print(p)  # 调用 __str__：Person: 张三
print(len(p))