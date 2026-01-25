# Python Multiprocessing 多进程并发编程指南

## 目录
1. [核心概念](#核心概念)
2. [进程 vs 线程](#进程-vs-线程)
3. [基础用法](#基础用法)
4. [进程间通信](#进程间通信)
5. [进程同步](#进程同步)
6. [最佳实践](#最佳实践)
7. [常见问题](#常见问题)

---

## 核心概念

### 什么是进程（Process）？

**进程**是操作系统中的一个独立执行单元，具有以下特点：

- **独立内存空间**：每个进程有自己的内存区域，进程间数据相互隔离
- **独立执行**：进程可以在多个 CPU 核心上并行执行
- **系统管理**：由操作系统负责进程的调度和管理

### multiprocessing 库的用途

Python 的 `multiprocessing` 库提供了：

- **进程创建与管理**：创建、启动、停止子进程
- **进程间通信（IPC）**：进程间数据传递
- **进程同步**：多进程协调执行
- **共享资源**：安全地在进程间共享数据

---

## 进程 vs 线程

### 对比表

| 特性 | 进程（Process） | 线程（Thread） |
|------|-----------------|----------------|
| **内存空间** | 完全独立 | 共享内存 |
| **创建开销** | 大（秒级） | 小（毫秒级） |
| **通信方式** | IPC（复杂） | 直接共享变量（需加锁） |
| **GIL 影响** | 无（真正并行） | 受限（伪并行） |
| **安全性** | 高（内存隔离） | 低（需细心同步） |
| **崩溃影响** | 只影响该进程 | 可能影响整个程序 |
| **适用场景** | CPU 密集型 | I/O 密集型 |

### GIL 是什么？

**GIL（Global Interpreter Lock）** 是 Python 解释器的全局锁，确保同一时间只有一个线程执行 Python 字节码。

- **线程受限**：多个线程无法真正并行执行 Python 代码
- **进程无限制**：每个进程有独立的 GIL，可真正并行

**举例：**
```python
# 使用线程计算 100 万个平方（伪并行）
# 实际时间 ≈ 单线程时间（GIL 限制）

# 使用进程计算 100 万个平方（真并行）
# 实际时间 ≈ 单进程时间 / 进程数（真正加速）
```

---

## 基础用法

### 1. 最简单的进程

```python
from multiprocessing import Process
import os

def worker(name):
    """子进程的执行函数"""
    print(f"子进程 {name} (PID: {os.getpid()}) 正在运行")

if __name__ == '__main__':
    print(f"主进程 (PID: {os.getpid()}) 启动")

    # 创建进程
    p = Process(target=worker, args=("Worker-1",))

    # 启动进程
    p.start()

    # 等待进程完成
    p.join()

    print("主进程完成")
```

**输出：**
```
主进程 (PID: 12345) 启动
子进程 Worker-1 (PID: 12346) 正在运行
主进程完成
```

### 2. 创建多个进程

```python
from multiprocessing import Process
import time

def worker(worker_id):
    """模拟耗时任务"""
    print(f"Worker {worker_id} 开始")
    time.sleep(2)
    print(f"Worker {worker_id} 完成")

if __name__ == '__main__':
    # 创建多个进程
    processes = []
    for i in range(4):
        p = Process(target=worker, args=(i,))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    print("所有进程已完成")
```

**执行时间分析：**
- 单进程：8 秒（4 个任务 × 2 秒）
- 4 进程：2 秒（4 个任务并行运行）

### 3. 进程类

```python
from multiprocessing import Process

class Worker(Process):
    """将进程定义为类"""

    def __init__(self, worker_id):
        super().__init__()
        self.worker_id = worker_id

    def run(self):
        """进程执行的方法"""
        print(f"Worker {self.worker_id} 执行中...")

if __name__ == '__main__':
    workers = [Worker(i) for i in range(3)]

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()
```

### 4. 进程池（Pool）

```python
from multiprocessing import Pool
import time

def task(x):
    """计算平方"""
    time.sleep(1)
    return x ** 2

if __name__ == '__main__':
    # 创建进程池，包含 4 个工作进程
    with Pool(processes=4) as pool:
        # 使用 map 函数并行处理数据
        results = pool.map(task, range(10))

    print(f"结果: {results}")
```

**Pool 的优点：**
- 自动管理进程生命周期
- 自动分配任务到可用进程
- 支持异步执行

---

## 进程间通信

### 1. Queue（队列）

**用途**：在进程间传递数据（FIFO）

```python
from multiprocessing import Process, Queue

def producer(q):
    """生产者进程"""
    for i in range(5):
        q.put(f"Item {i}")
    print("生产者完成")

def consumer(q):
    """消费者进程"""
    while True:
        item = q.get()
        if item is None:
            break
        print(f"消费: {item}")

if __name__ == '__main__':
    q = Queue()

    # 启动生产者和消费者
    p1 = Process(target=producer, args=(q,))
    p2 = Process(target=consumer, args=(q,))

    p1.start()
    p2.start()

    p1.join()
    q.put(None)  # 信号停止
    p2.join()
```

**Queue 的常用方法：**
```python
q = Queue()
q.put(data)           # 放入数据
data = q.get()        # 取出数据（阻塞式）
data = q.get_nowait() # 非阻塞式，队列空则抛异常
q.empty()             # 检查是否空
q.full()              # 检查是否满
q.qsize()             # 获取队列大小
```

### 2. Pipe（管道）

**用途**：两个进程间的双向通信

```python
from multiprocessing import Process, Pipe

def sender(conn):
    """发送者"""
    conn.send("Hello from process 1")
    conn.send([1, 2, 3])
    conn.close()

def receiver(conn):
    """接收者"""
    msg1 = conn.recv()
    msg2 = conn.recv()
    print(f"收到: {msg1}")
    print(f"收到: {msg2}")
    conn.close()

if __name__ == '__main__':
    # 创建管道，返回两个连接端点
    conn1, conn2 = Pipe()

    p1 = Process(target=sender, args=(conn1,))
    p2 = Process(target=receiver, args=(conn2,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
```

**Pipe vs Queue：**
- **Pipe**：点对点通信（2 个进程）
- **Queue**：多生产者-多消费者（N 个进程）

### 3. Manager（数据共享）

**用途**：在进程间共享 Python 对象（dict, list 等）

```python
from multiprocessing import Process, Manager

def increment_counter(shared_dict):
    """增加计数器"""
    for i in range(100):
        shared_dict['count'] += 1

if __name__ == '__main__':
    with Manager() as manager:
        # 创建共享字典
        shared_dict = manager.dict()
        shared_dict['count'] = 0

        # 创建多个进程修改共享数据
        processes = []
        for i in range(4):
            p = Process(target=increment_counter, args=(shared_dict,))
            processes.append(p)
            p.start()

        # 等待所有进程完成
        for p in processes:
            p.join()

        print(f"最终计数: {shared_dict['count']}")
```

**Manager 支持的类型：**
- `manager.dict()`：共享字典
- `manager.list()`：共享列表
- `manager.Queue()`：管理的队列
- `manager.Lock()`：管理的锁
- `manager.Semaphore()`：管理的信号量

**注意**：Manager 的开销较大，仅在必要时使用。

---

## 进程同步

### 1. Lock（互斥锁）

**用途**：防止多个进程同时访问临界区

```python
from multiprocessing import Process, Lock

def write_file(lock, file_name, content):
    """线程安全地写文件"""
    with lock:
        with open(file_name, 'a') as f:
            f.write(content + '\n')

if __name__ == '__main__':
    lock = Lock()

    processes = []
    for i in range(5):
        p = Process(target=write_file, args=(lock, "output.txt", f"Line {i}"))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
```

**Lock 的方法：**
```python
lock = Lock()
lock.acquire()      # 获取锁（阻塞式）
lock.release()      # 释放锁
with lock:          # 上下文管理器（推荐）
    # 临界区代码
    pass
```

### 2. RLock（可重入锁）

**用途**：允许同一进程多次获取锁

```python
from multiprocessing import RLock

rlock = RLock()

def recursive_function():
    with rlock:
        print("第一层")
        with rlock:  # 同一进程可以再次获取
            print("第二层")
```

### 3. Event（事件）

**用途**：进程间的信号同步

```python
from multiprocessing import Process, Event
import time

def waiter(event):
    """等待事件发生"""
    print("等待者: 等待事件...")
    event.wait()  # 阻塞直到事件被设置
    print("等待者: 事件已发生!")

def setter(event):
    """设置事件"""
    time.sleep(2)
    print("设置者: 设置事件")
    event.set()

if __name__ == '__main__':
    event = Event()

    p1 = Process(target=waiter, args=(event,))
    p2 = Process(target=setter, args=(event,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
```

**Event 的方法：**
```python
event = Event()
event.set()      # 设置事件标志为 True
event.clear()    # 清除事件标志为 False
event.is_set()   # 检查标志状态
event.wait()     # 阻塞直到标志为 True
```

### 4. Semaphore（信号量）

**用途**：限制同时访问资源的进程数

```python
from multiprocessing import Process, Semaphore
import time

def worker(semaphore, worker_id):
    """同时最多 2 个工作进程访问资源"""
    with semaphore:
        print(f"Worker {worker_id} 获得资源")
        time.sleep(2)
        print(f"Worker {worker_id} 释放资源")

if __name__ == '__main__':
    # 创建信号量，初始值为 2
    semaphore = Semaphore(2)

    processes = []
    for i in range(5):
        p = Process(target=worker, args=(semaphore, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
```

### 5. Condition（条件变量）

**用途**：等待某个条件成立

```python
from multiprocessing import Process, Condition
import time

def producer(condition):
    """生产者"""
    with condition:
        time.sleep(2)
        print("生产者: 生产数据")
        condition.notify_all()  # 唤醒所有等待者

def consumer(condition, consumer_id):
    """消费者"""
    with condition:
        print(f"消费者 {consumer_id}: 等待数据...")
        condition.wait()  # 等待被唤醒
        print(f"消费者 {consumer_id}: 消费数据")

if __name__ == '__main__':
    condition = Condition()

    p_producer = Process(target=producer, args=(condition,))

    consumers = []
    for i in range(3):
        p = Process(target=consumer, args=(condition, i))
        consumers.append(p)

    p_producer.start()
    for p in consumers:
        p.start()

    p_producer.join()
    for p in consumers:
        p.join()
```

---

## 最佳实践

### 1. 进程启动的保护

**❌ 错误做法：**
```python
from multiprocessing import Process

def worker():
    print("working...")

# 没有 if __name__ == '__main__' 保护
p = Process(target=worker)
p.start()
```

**✅ 正确做法：**
```python
from multiprocessing import Process

def worker():
    print("working...")

if __name__ == '__main__':
    p = Process(target=worker)
    p.start()
    p.join()
```

**原因**：在 Windows 和某些情况下，模块会被重新导入，导致无限递归创建进程。

### 2. 异常处理

```python
from multiprocessing import Process
import traceback

def worker():
    try:
        # 可能出错的代码
        result = 10 / int("abc")
    except Exception as e:
        print(f"错误发生: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    p = Process(target=worker)
    p.start()
    p.join()

    if p.exitcode != 0:
        print("子进程异常退出")
```

### 3. 进程优雅终止

```python
from multiprocessing import Process
import time

def long_running_task(stop_event):
    """可以被停止的任务"""
    while not stop_event.is_set():
        print("任务运行中...")
        time.sleep(1)
    print("任务已停止")

if __name__ == '__main__':
    from multiprocessing import Event

    stop_event = Event()
    p = Process(target=long_running_task, args=(stop_event,))
    p.start()

    time.sleep(5)
    stop_event.set()  # 通知进程停止
    p.join(timeout=5)  # 等待最多 5 秒

    if p.is_alive():
        p.terminate()  # 强制终止
        p.join()
```

### 4. 使用 with 语句

```python
from multiprocessing import Pool

def task(x):
    return x ** 2

if __name__ == '__main__':
    # 推荐：自动清理资源
    with Pool(processes=4) as pool:
        results = pool.map(task, range(10))
```

### 5. 选择合适的启动方法

```python
from multiprocessing import get_context

if __name__ == '__main__':
    # spawn：启动完全独立的 Python 解释器（最安全）
    ctx = get_context('spawn')
    p = ctx.Process(target=worker)

    # fork：复制父进程（Linux/Unix，速度快但可能不安全）
    ctx = get_context('fork')
    p = ctx.Process(target=worker)

    # forkserver：启动服务器进程（平衡方案）
    ctx = get_context('forkserver')
    p = ctx.Process(target=worker)
```

---

## 常见问题

### Q1：什么时候用进程，什么时候用线程？

**使用进程的场景：**
- CPU 密集型任务（计算、数据处理）
- 需要真正并行执行
- 进程间隔离很重要

**使用线程的场景：**
- I/O 密集型任务（网络请求、文件读写）
- 进程间需要频繁共享数据
- 资源使用受限

### Q2：进程过多会怎样？

**问题：**
```python
# ❌ 创建过多进程
for i in range(10000):
    p = Process(target=worker)
    p.start()
```

**后果：**
- 内存占用过高
- CPU 上下文切换开销大
- 系统响应变慢

**建议：**
- 通常不超过 CPU 核心数
- 使用进程池而不是手动创建

### Q3：子进程无法访问父进程的变量？

**❌ 错误：**
```python
global_var = 10

def worker():
    print(global_var)  # 看不到父进程的修改

global_var = 20
p = Process(target=worker)
p.start()
```

**✅ 解决方案 1：传参**
```python
def worker(value):
    print(value)

p = Process(target=worker, args=(20,))
p.start()
```

**✅ 解决方案 2：使用 Manager**
```python
from multiprocessing import Manager

if __name__ == '__main__':
    with Manager() as manager:
        shared_dict = manager.dict()
        shared_dict['var'] = 20
        p = Process(target=worker, args=(shared_dict,))
        p.start()
```

### Q4：如何在进程间传递复杂对象？

```python
from multiprocessing import Queue
import pickle

class ComplexObject:
    def __init__(self, data):
        self.data = data

if __name__ == '__main__':
    q = Queue()

    # 可以直接传递（pickle 序列化）
    obj = ComplexObject([1, 2, 3])
    q.put(obj)

    received_obj = q.get()
    print(received_obj.data)
```

**注意**：对象必须是可序列化的（picklable）。

### Q5：进程间通信的性能如何？

**IPC 性能排序（从快到慢）：**
1. 共享内存（SharedMemory）- 最快
2. Pipe - 快
3. Queue - 中等
4. Manager - 较慢（网络通信）

**选择建议：**
- 大数据传输：使用 SharedMemory
- 简单数据传输：使用 Pipe
- 多生产者-消费者：使用 Queue
- 需要特殊数据结构：使用 Manager

---

## 总结

| 功能 | 工具 | 用途 |
|------|------|------|
| **创建进程** | `Process` | 创建单个进程 |
| **进程池** | `Pool` | 管理多个进程 |
| **数据传递** | `Queue`, `Pipe` | 进程间通信 |
| **数据共享** | `Manager` | 共享 Python 对象 |
| **同步** | `Lock`, `Event`, `Semaphore` | 进程协调 |

**核心原则：**
1. ✅ 使用 `if __name__ == '__main__'` 保护
2. ✅ 合理选择 IPC 方式
3. ✅ 适当使用同步机制
4. ✅ 优雅处理进程退出
5. ✅ 监控进程资源使用

