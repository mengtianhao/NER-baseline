import torch
import os

class Model_config:
    def __init__(self, task_name, model_name="chinese-bert-wwm-ext"):
        # 项目路径
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 预训练模型路径
        self.model_dir = os.path.join(self.project_dir, 'data', 'model_data')
        self.data_dir = "CBLUEDatasets"
        self.task_name = task_name
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        if 'albert' in model_name:
            self.model_type = 'albert'
        else:
            self.model_type = "bert"
        #
        self.output_dir = "data/output"
        self.result_output_dir = "data/result_output"
        self.max_length = 128
        self.train_batch_size = 2
        self.eval_batch_size = 16
        self.learning_rate = 3e-5
        self.epochs = 5
        self.warmup_proportion = 0.1
        self.earlystop_patience = 100
        self.max_grad_norm = 0.0
        self.logging_steps = 200
        self.save_steps = 200
        self.seed = 2021
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8