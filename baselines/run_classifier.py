import os
import sys
from transformers import BertTokenizer, BertForTokenClassification, AlbertForTokenClassification
from cblue.data.data_process import DataProcessor
from cblue.data.dataset import Dataset
from cblue.trainer.train import MyTrainer
from cblue.utils import init_logger, seed_everything
from cblue.models import ZenNgramDict, ZenForTokenClassification
from transformers import logging
logging.set_verbosity_error()
sys.path.append('.')


def main(config, do_train=False, do_predict=False,):
    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)
    output_dir = os.path.join(config.output_dir, config.model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(config.result_output_dir):
        os.mkdir(config.result_output_dir)

    logger = init_logger(os.path.join(output_dir, f'{config.task_name}_{config.model_name}.log'))

    seed_everything(config.seed)

    dataset_class, data_processor_class = Dataset, DataProcessor

    if config.model_type == 'bert':
        tokenizer_class, model_class = BertTokenizer, BertForTokenClassification
    elif config.model_type == 'roberta':
        tokenizer_class, model_class = BertTokenizer, BertForTokenClassification
    elif config.model_type == 'albert':
        tokenizer_class, model_class = BertTokenizer, AlbertForTokenClassification
    elif config.model_type == 'zen':
        tokenizer_class, model_class = BertTokenizer, ZenForTokenClassification

    if do_train:
        tokenizer = tokenizer_class.from_pretrained(os.path.join(config.model_dir, config.model_name))

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

        model = model_class.from_pretrained(os.path.join(config.model_dir, config.model_name), num_labels=data_processor.num_labels)

        trainer = MyTrainer(config=config, model=model, data_processor=data_processor,
                            tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
                            logger=logger, model_class=model_class, ngram_dict=ngram_dict)

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
            
        model = model_class.from_pretrained(output_dir, num_labels=data_processor.num_labels)
        trainer = MyTrainer(config=config, model=model, data_processor=data_processor,
                            tokenizer=tokenizer, logger=logger, model_class=model_class, ngram_dict=ngram_dict)
        trainer.predict(test_dataset=test_dataset, model=model)


if __name__ == '__main__':
    main()
