代码语言模型指令微调
==========================

简介
----

本模块用于对代码语言模型做指令式微调（Instruction Fine-tuning），使模型更好地遵循自然语言的指令（例如“给出函数注释”、“补全 TODO” 等）并提升交互式代码生成/修改能力。

功能展示
--------

- 将任务数据（instruction, input, output）格式化为训练样本。
- 支持常见的训练流水线（数据并行、分布式训练、混合精度）。
- 提供验证/评估脚本（基于 BLEU / ROUGE / pass@k / 人工示例集等）。

示例配置（简要）
----------------

.. code-block:: json

   {
     "model": "codellama-7b",
     "train_data": "data/instructions/train.jsonl",
     "batch_size": 8,
     "lr": 2e-5
   }

使用教程
--------

1. 数据准备：将数据组织为（instruction, input, output）三元组，推荐 JSONL。
2. 启动训练：
   .. code-block:: shell

      CUDA_VISIBLE_DEVICES=0,1 python -m ncc.train.instruction_finetune -c config/if_config.json

3. 验证与导出：训练完成后运行评估脚本并导出 checkpoint。
