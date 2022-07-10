import os
import sys

import torch
from transformers import BertTokenizer, BertForTokenClassification, AlbertForTokenClassification, BertModel
from cblue.data.data_process import DataProcessor
from cblue.data.dataset import Dataset
from cblue.trainer.train import MyTrainer
from cblue.utils import init_logger, seed_everything
from cblue.models import ZenNgramDict, ZenForTokenClassification
from transformers import logging

logging.set_verbosity_error()
sys.path.append('.')


def main(config, do_train=False, do_predict=False, ):
    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)
    output_dir = os.path.join(config.output_dir, config.classify_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(config.output_dir, config.classify_name, config.model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(config.result_output_dir):
        os.mkdir(config.result_output_dir)
    # print(output_dir)

    logger = init_logger(os.path.join(output_dir, f'{config.task_name}_{config.model_name}.log'))

    seed_everything(config.seed)

    dataset_class, data_processor_class = Dataset, DataProcessor

    if config.model_type == 'bert' and config.classify_name == "BertForTokenClassification":
        tokenizer_class, model_class = BertTokenizer, BertForTokenClassification
    elif config.model_type == 'bert' and config.classify_name == "CRF":
        tokenizer_class, model_class = BertTokenizer, BertModel
    elif config.model_type == 'roberta' and config.classify_name == "BertForTokenClassification":
        tokenizer_class, model_class = BertTokenizer, BertForTokenClassification
    elif config.model_type == 'roberta' and config.classify_name == "CRF":
        tokenizer_class, model_class = BertTokenizer, BertModel
    elif config.model_type == 'albert':
        tokenizer_class, model_class = BertTokenizer, AlbertForTokenClassification
    elif config.model_type == 'zen':
        tokenizer_class, model_class = BertTokenizer, ZenForTokenClassification

    if do_train:
        model_save_dir = os.path.join(config.output_dir, config.classify_name, config.model_name)
        model_save_path = os.path.join(model_save_dir, 'pytorch_model.bin')
        best_score = 0.0
        if os.path.exists(model_save_path):
            model_file_path = model_save_dir
            checkpoint = torch.load(os.path.join(model_save_dir, 'training_config.bin'))
            best_score = checkpoint['best_score']  # 加载上次的最好的正确率
        else:
            model_file_path = os.path.join(config.model_dir, config.model_name)
        tokenizer = tokenizer_class.from_pretrained(model_file_path)

        # compatible with 'ZEN' model
        ngram_dict = None
        if config.model_type == 'zen':
            ngram_dict = ZenNgramDict(os.path.join(config.model_dir, config.model_name), tokenizer=tokenizer)

        data_processor = data_processor_class(root=config.data_dir)
        train_samples = data_processor.get_train_sample()
        eval_samples = data_processor.get_dev_sample()

        train_dataset = dataset_class(train_samples, data_processor, tokenizer, mode='train',
                                      model_type=config.model_type, ngram_dict=ngram_dict, max_length=config.max_length)
        eval_dataset = dataset_class(eval_samples, data_processor, tokenizer, mode='eval',
                                     model_type=config.model_type, ngram_dict=ngram_dict, max_length=config.max_length)

        if config.classify_name != "CRF":
            model = model_class.from_pretrained(model_file_path, num_labels=data_processor.num_labels)
        else:
            model = model_class.from_pretrained(model_file_path)

        trainer = MyTrainer(config=config, model=model, data_processor=data_processor,
                            tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
                            logger=logger, model_class=model_class, ngram_dict=ngram_dict,
                            num_labels=data_processor.num_labels, best_score=best_score)

        global_step, best_step = trainer.train()

    if do_predict:
        tokenizer = tokenizer_class.from_pretrained(output_dir)

        ngram_dict = None
        if config.model_type == 'zen':
            ngram_dict = ZenNgramDict(os.path.join(config.model_dir, config.model_name), tokenizer=tokenizer)

        data_processor = data_processor_class(root=config.data_dir)
        test_samples = data_processor.get_test_sample()

        test_dataset = dataset_class(test_samples, data_processor, tokenizer, mode='test', ngram_dict=ngram_dict,
                                     max_length=config.max_length, model_type=config.model_type)

        if config.classify_name != "CRF":
            model = model_class.from_pretrained(output_dir, num_labels=data_processor.num_labels)
        else:
            model = model_class.from_pretrained(output_dir)

        trainer = MyTrainer(config=config, model=model, data_processor=data_processor,
                            tokenizer=tokenizer, logger=logger, model_class=model_class, ngram_dict=ngram_dict,
                            num_labels=data_processor.num_labels)
        trainer.predict(test_dataset=test_dataset, model=model)


if __name__ == '__main__':
    main()
