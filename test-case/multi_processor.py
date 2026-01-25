

import multiprocessing
def worker(num):
    """多进程工作函数"""
    print(f"Worker: {num}")


if __name__ == "__main__":
    
    processes = []

    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=worker, args=(1,))
    p.start()

    print(p.is_alive())
    print(p.pid)

    p.join()



    
    

