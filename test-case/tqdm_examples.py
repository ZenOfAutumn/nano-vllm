"""
tqdm 进度条库的各种使用示例
展示 tqdm 的基础用法到高级用法
"""

import time

from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto


def example1_basic_progress_bar():
    """示例 1: 最简单的进度条"""
    print("\n=== 示例 1: 最简单的进度条 ===")
    for i in tqdm(range(10)):
        time.sleep(0.5)


def example2_with_description():
    """示例 2: 显示说明文字"""
    print("\n=== 示例 2: 显示说明文字 ===")
    for i in tqdm(range(10), desc="处理数据"):
        time.sleep(0.5)


def example3_manual_update():
    """示例 3: 手动更新进度条"""
    print("\n=== 示例 3: 手动更新进度条 ===")
    pbar = tqdm(total=100, desc="下载文件")

    for i in range(10):
        time.sleep(0.1)
        pbar.update(10)  # 每次增加 10

    pbar.close()


def example4_with_context_manager():
    """示例 4: 使用 with 语句"""
    print("\n=== 示例 4: 使用 with 语句 ===")
    with tqdm(total=100) as pbar:
        for i in range(10):
            time.sleep(0.1)
            pbar.update(10)


def example5_file_processing():
    """示例 5: 文件处理进度"""
    print("\n=== 示例 5: 文件处理进度 ===")
    files = ['file1.txt', 'file2.txt', 'file3.txt', 'file4.txt', 'file5.txt']

    for filename in tqdm(files, desc="处理文件"):
        time.sleep(0.5)
        print(f"\n  处理 {filename}")


def example6_list_comprehension():
    """示例 6: 列表推导式"""
    print("\n=== 示例 6: 列表推导式 ===")
    results = [x**2 for x in tqdm(range(50), desc="计算平方")]
    print(f"计算完成，前 10 个结果: {results[:10]}")


def example7_custom_unit():
    """示例 7: 自定义单位"""
    print("\n=== 示例 7: 自定义单位（字节） ===")
    bytes_total = 1024 * 1024  # 1 MB

    with tqdm(total=bytes_total, unit='B', unit_scale=True, desc="下载") as pbar:
        for i in range(10):
            pbar.update(100 * 1024)
            time.sleep(0.1)


def example8_dynamic_update():
    """示例 8: 动态更新描述"""
    print("\n=== 示例 8: 动态更新描述 ===")
    pbar = tqdm(total=100, desc="训练")

    for i in range(100):
        pbar.set_description(f"训练 epoch {i // 20}")
        pbar.update(1)
        time.sleep(0.01)

    pbar.close()


def example9_custom_postfix():
    """示例 9: 自定义显示信息"""
    print("\n=== 示例 9: 自定义显示信息（set_postfix） ===")
    pbar = tqdm(total=100, desc="训练")

    for i in range(100):
        loss = 1.5 - i * 0.01
        pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'epoch': i // 25,
            'lr': f'{0.001:.5f}'
        })
        pbar.update(1)
        time.sleep(0.01)

    pbar.close()


def example10_nested_progress_bars():
    """示例 10: 嵌套进度条"""
    print("\n=== 示例 10: 嵌套进度条 ===")
    for i in tqdm(range(5), desc="外层循环", position=0):
        for j in tqdm(range(10), desc="内层循环", position=1, leave=False):
            time.sleep(0.05)


def example11_progress_bar_parameters():
    """示例 11: 各种参数组合"""
    print("\n=== 示例 11: 各种参数组合 ===")

    # 参数 1：ncols 设置宽度
    print("宽进度条:")
    for i in tqdm(range(20), ncols=100, desc="宽"):
        time.sleep(0.1)

    # 参数 2：leave 控制是否保留进度条
    print("\n不保留进度条:")
    for i in tqdm(range(20), desc="临时", leave=False):
        time.sleep(0.05)

    print("进度条已消失")


def example12_without_iterable():
    """示例 12: 不基于可迭代对象的进度条"""
    print("\n=== 示例 12: 手动指定总数的进度条 ===")
    pbar = tqdm(total=100, desc="处理")

    items_processed = 0
    while items_processed < 100:
        # 模拟处理
        time.sleep(0.05)
        items_processed += 10
        pbar.update(10)

    pbar.close()


def example13_simulation_like_nano_vllm():
    """示例 13: 模拟 nano-vllm 中的用法"""
    print("\n=== 示例 13: 模拟 nano-vllm 中的用法 ===")

    prompts = [f"prompt_{i}" for i in range(5)]

    # 初始化进度条
    pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

    # 模拟推理循环
    prefill_throughput = 0
    decode_throughput = 0

    for seq_id in range(len(prompts)):
        time.sleep(0.3)

        # 模拟 Prefill 和 Decode 阶段的吞吐量
        if seq_id < 2:
            prefill_throughput = 1000 + seq_id * 100
        else:
            decode_throughput = 500 + seq_id * 50

        # 更新进度条显示吞吐量
        pbar.set_postfix({
            "Prefill": f"{int(prefill_throughput)}tok/s",
            "Decode": f"{int(decode_throughput)}tok/s",
        })

        # 更新进度
        pbar.update(1)

    pbar.close()


def example14_try_tqdm_auto():
    """示例 14: 使用 tqdm.auto（自动选择显示方式）"""
    print("\n=== 示例 14: 使用 tqdm.auto ===")
    print("在 Jupyter 中显示漂亮的进度条，在终端显示普通进度条")

    for i in tqdm_auto(range(20), desc="自动选择"):
        time.sleep(0.1)


def main():
    """运行所有示例"""
    print("=" * 60)
    print("tqdm 进度条库的完整示例演示")
    print("=" * 60)

    # 可以选择运行单个或全部示例
    examples = [
        ("基础进度条", example1_basic_progress_bar),
        ("带说明文字", example2_with_description),
        ("手动更新", example3_manual_update),
        ("with 语句", example4_with_context_manager),
        ("文件处理", example5_file_processing),
        ("列表推导式", example6_list_comprehension),
        ("自定义单位", example7_custom_unit),
        ("动态更新描述", example8_dynamic_update),
        ("自定义信息", example9_custom_postfix),
        ("嵌套进度条", example10_nested_progress_bars),
        ("参数组合", example11_progress_bar_parameters),
        ("手动模式", example12_without_iterable),
        ("nano-vllm 模拟", example13_simulation_like_nano_vllm),
        ("tqdm.auto", example14_try_tqdm_auto),
    ]

    print("\n可用的示例:")
    for idx, (name, _) in enumerate(examples, 1):
        print(f"{idx}. {name}")

    print("\n" + "=" * 60)

    # 运行所有示例
    user_input = input("\n是否运行所有示例? (y/n，默认 n): ").strip().lower()

    if user_input == 'y':
        for name, example_func in examples:
            try:
                example_func()
            except Exception as e:
                print(f"错误: {e}")
            time.sleep(0.5)
    else:
        # 交互式选择
        while True:
            try:
                choice = input("\n输入示例编号 (1-14)，或 'q' 退出: ").strip().lower()

                if choice == 'q':
                    print("退出程序")
                    break

                idx = int(choice) - 1
                if 0 <= idx < len(examples):
                    name, example_func = examples[idx]
                    print(f"\n运行: {name}")
                    example_func()
                else:
                    print("无效的选择，请输入 1-14 之间的数字")
            except ValueError:
                print("无效的输入")

    print("\n" + "=" * 60)
    print("所有示例演示完成!")


if __name__ == '__main__':
    main()

