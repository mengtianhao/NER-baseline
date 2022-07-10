import torch
import os


class Model_config:
    def __init__(self, task_name, classify_name, model_name="chinese-bert-wwm-ext"):
        # 项目路径
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 预训练模型路径
        self.model_dir = os.path.join(self.project_dir, 'data', 'model_data')
        # 数据集目录
        self.data_dir = "Datasets"
        # 分类器名称
        self.classify_name = classify_name
        # 任务名称
        self.task_name = task_name
        # 运行设备
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 预训练模型名称
        self.model_name = model_name
        if 'albert' in model_name:
            self.model_type = 'albert'
        else:
            self.model_type = "bert"
        # 模型保存路径
        self.output_dir = os.path.join(self.project_dir, 'data', 'output')
        # 测试集结果保存路径
        self.result_output_dir = os.path.join(self.project_dir, 'data', 'result_output')
        # 句子的最大长度
        self.max_length = 128
        # 训练批次大小
        self.train_batch_size = 64
        # 验证批次大小
        self.eval_batch_size = 16
        # 初始化学习率
        self.learning_rate = 3e-5
        # 训练的epochs
        self.epochs = 1
        # Warmup预热学习率
        self.warmup_proportion = 0.1
        # 自上次模型在验证集上损失降低之后等待的时间
        self.earlystop_patience = 100
        #
        self.max_grad_norm = 0.0
        # 每200步输出一次logging
        self.logging_steps = 100
        # 每200步保存一下模型
        self.save_steps = 100
        # 随机种子
        self.seed = 2021
        # 权值衰减
        self.weight_decay = 0.01
        # 为了增加数值计算的稳定性而加到分母里的项
        self.adam_epsilon = 1e-8
        # 隐藏层维度
        self.hidden_size = 768