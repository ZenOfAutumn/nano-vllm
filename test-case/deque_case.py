
from collections import deque

x = deque(range(10))

print(x)
x.append(11)
print(x)
y = x.pop()
print(x)


x.appendleft(0)
print(x)
x.popleft()
print(x)


print("=====")


z: deque[int] = deque(range(10))
print(z)

print(z[0])
print(z)
print(z[-1])
print(z)

