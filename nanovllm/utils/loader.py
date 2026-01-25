# ============ 导入模块 ============
# 操作系统相关功能：用于文件路径操作
import os
# glob 模块：用于文件路径模式匹配和通配符扩展
from glob import glob

# PyTorch 主模块：张量和自动梯度计算
import torch
# safetensors 模块：安全地加载和保存张量数据的库
from safetensors import safe_open
# PyTorch 神经网络模块：神经网络层和模型基类
from torch import nn


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """
    默认的权重加载器：将加载的权重复制到模型参数中。

    这是一个简单的权重转移函数，用于那些没有自定义 weight_loader 方法的参数。

    参数：
        param (nn.Parameter): 模型中的参数对象（需要被更新的目标参数）
        loaded_weight (torch.Tensor): 从文件中加载的权重张量（源数据）
    """
    # 使用 copy_ 方法将加载的权重复制到参数中（原地操作，不创建新的张量）
    # copy_ 是 PyTorch 中的原地操作，使用下划线后缀表示
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    加载模型权重的主函数。

    从指定目录中的 safetensors 文件加载权重到模型中。
    支持两种加载方式：
    1. 针对已打包模块的特殊加载（使用 packed_modules_mapping）
    2. 直接加载到对应参数的标准方式

    参数：
        model (nn.Module): 要加载权重的模型对象
        path (str): 包含 safetensors 文件的目录路径
    """
    # ============ 步骤 1：获取已打包模块的映射关系 ============
    # 尝试从模型中获取 packed_modules_mapping 属性
    # 如果不存在则使用空字典 {} 作为默认值
    # packed_modules_mapping 用于将权重名称映射到实际的参数名称和分片 ID
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # ============ 步骤 2：遍历所有 safetensors 文件 ============
    # glob() 会返回所有匹配模式的文件路径列表
    # os.path.join() 构建完整路径：path + "*.safetensors"
    # "*.safetensors" 是文件名匹配模式，* 表示任意文件名
    for file in glob(os.path.join(path, "*.safetensors")):
        # ============ 步骤 3：打开 safetensors 文件 ============
        # safe_open() 以安全的方式打开 safetensors 文件
        # 参数说明：
        #   - file: 要打开的文件路径
        #   - "pt": 表示 PyTorch 格式（与 safetensors 兼容）
        #   - "cpu": 指定在 CPU 上加载张量（而不是 GPU）
        # with 语句确保文件被正确关闭
        with safe_open(file, "pt", "cpu") as f:
            # ============ 步骤 4：遍历文件中的所有权重 ============
            # f.keys() 返回文件中所有权重张量的名称列表
            for weight_name in f.keys():
                # ============ 步骤 5：检查是否需要特殊处理（已打包模块） ============
                # 遍历已打包模块的映射关系
                for k in packed_modules_mapping:
                    # 如果当前权重名称包含某个打包模块的键 k
                    if k in weight_name:
                        # 从映射中获取对应的值：(真实参数名, 分片ID)
                        # v: 真实的参数名称
                        # shard_id: 用于张量并行的分片 ID
                        v, shard_id = packed_modules_mapping[k]
                        # 将权重名称中的 k 替换为 v（从文件中的名称转换为模型中的参数名）
                        param_name = weight_name.replace(k, v)
                        # 从模型中获取对应参数的引用
                        param = model.get_parameter(param_name)
                        # 从参数中获取自定义的 weight_loader 方法
                        weight_loader = getattr(param, "weight_loader")
                        # 调用参数的 weight_loader 方法加载权重
                        # 这支持自定义的权重加载逻辑（如量化、转换等）
                        # 传入三个参数：参数对象、加载的权重、分片 ID
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        # break 跳出内层 for 循环（已处理该权重，无需继续检查其他映射）
                        break
                # ============ 步骤 6：处理非打包模块的权重 ============
                # else 子句与 for 循环关联：当循环正常结束（没有 break）时执行
                # 即：权重名称不在任何已打包模块中，使用标准加载方式
                else:
                    # 从模型中获取权重对应的参数
                    param = model.get_parameter(weight_name)
                    # 尝试从参数中获取 weight_loader 方法
                    # 如果参数没有自定义 weight_loader，则使用 default_weight_loader
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    # 调用 weight_loader 方法加载权重
                    # 注意：这里只传两个参数（没有 shard_id），因为是非打包模块
                    # default_weight_loader 只需要参数和权重两个参数
                    weight_loader(param, f.get_tensor(weight_name))
