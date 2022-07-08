import os
import sys
sys.path.append('.')
from transformers import BertTokenizer, BertForSequenceClassification, AlbertForSequenceClassification, \
    BertForTokenClassification, AlbertForTokenClassification
from cblue.data.data_process import EEDataProcessor
from cblue.data.dataset import EEDataset
from cblue.trainer.train import EETrainer
from cblue.utils import init_logger, seed_everything
from cblue.models import ZenNgramDict, ZenForSequenceClassification, ZenForTokenClassification
from transformers import logging
logging.set_verbosity_error()


MODEL_CLASS = {
    'bert': (BertTokenizer, BertForSequenceClassification),
    'roberta': (BertTokenizer, BertForSequenceClassification),
    'albert': (BertTokenizer, AlbertForSequenceClassification),
    'zen': (BertTokenizer, ZenForSequenceClassification)
}

TOKEN_MODEL_CLASS = {
    'bert': (BertTokenizer, BertForTokenClassification),
    'roberta': (BertTokenizer, BertForTokenClassification),
    'albert': (BertTokenizer, AlbertForTokenClassification),
    'zen': (BertTokenizer, ZenForTokenClassification)
}


def main(config, do_train=False, do_predict=False,):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", default=None, type=str, required=True,
    #                     help="The task data directory.")
    # parser.add_argument("--model_dir", default=None, type=str, required=True,
    #                     help="The directory of pretrained models")
    # parser.add_argument("--model_type", default=None, type=str, required=True,
    #                     help="The type of selected pretrained models.")
    # parser.add_argument("--model_name", default=None, type=str, required=True,
    #                     help="The path of selected pretrained models. (e.g. chinese-bert-wwm)")
    # parser.add_argument("--task_name", default=None, type=str, required=True,
    #                     help="The name of task to train")
    # parser.add_argument("--output_dir", default=None, type=str, required=True,
    #                     help="The path of result data and models to be saved.")
    # parser.add_argument("--do_train", action='store_true',
    #                     help="Whether to run training.")
    # parser.add_argument("--do_predict", action='store_true',
    #                     help="Whether to run the models in inference mode on the test set.")
    # parser.add_argument("--result_output_dir", default=None, type=str, required=True,
    #                     help="the directory of commit result to be saved")
    #
    # # models param
    # parser.add_argument("--max_length", default=128, type=int,
    #                     help="the max length of sentence.")
    # parser.add_argument("--train_batch_size", default=8, type=int,
    #                     help="Batch size for training.")
    # parser.add_argument("--eval_batch_size", default=8, type=int,
    #                     help="Batch size for evaluation.")
    # parser.add_argument("--learning_rate", default=5e-5, type=float,
    #                     help="The initial learning rate for Adam.")
    # parser.add_argument("--weight_decay", default=0.01, type=float,
    #                     help="Weight deay if we apply some.")
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float,
    #                     help="Epsilon for Adam optimizer.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float,
    #                     help="Max gradient norm.")
    # parser.add_argument("--epochs", default=3, type=int,
    #                     help="Total number of training epochs to perform.")
    # parser.add_argument("--warmup_proportion", default=0.1, type=float,
    #                     help="Proportion of training to perform linear learning rate warmup for, "
    #                          "E.g., 0.1 = 10% of training.")
    # parser.add_argument("--earlystop_patience", default=2, type=int,
    #                     help="The patience of early stop")
    #
    # parser.add_argument('--logging_steps', type=int, default=10,
    #                     help="Log every X updates steps.")
    # parser.add_argument('--save_steps', type=int, default=1000,
    #                     help="Save checkpoint every X updates steps.")
    # parser.add_argument('--seed', type=int, default=2021,
    #                     help="random seed for initialization")
    #
    # args = parser.parse_args()

    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)
    output_dir = os.path.join(config.output_dir, config.task_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir, config.model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(config.result_output_dir):
        os.mkdir(config.result_output_dir)

    logger = init_logger(os.path.join(output_dir, f'{config.task_name}_{config.model_name}.log'))
    seed_everything(config.seed)
    tokenizer_class, model_class = MODEL_CLASS[config.model_type]
    dataset_class, data_processor_class = (EEDataset, EEDataProcessor)
    trainer_class = EETrainer

    if config.task_name == 'ee':
        tokenizer_class, model_class = TOKEN_MODEL_CLASS[config.model_type]

    # logger.info("Training/evaluation parameters %s", args)
    if do_train:
        tokenizer = tokenizer_class.from_pretrained(os.path.join(config.model_dir, config.model_name))

        # compatible with 'ZEN' model
        ngram_dict = None
        if config.model_type == 'zen':
            ngram_dict = ZenNgramDict(os.path.join(config.model_dir, config.model_name), tokenizer=tokenizer)

        data_processor = data_processor_class(root=config.data_dir)
        train_samples = data_processor.get_train_sample()
        eval_samples = data_processor.get_dev_sample()

        if config.task_name == 'ee':
            train_dataset = dataset_class(train_samples, data_processor, tokenizer, mode='train',
                                          model_type=config.model_type, ngram_dict=ngram_dict, max_length=config.max_length)
            eval_dataset = dataset_class(eval_samples, data_processor, tokenizer, mode='eval',
                                         model_type=config.model_type, ngram_dict=ngram_dict, max_length=config.max_length)
        else:
            train_dataset = dataset_class(train_samples, data_processor, mode='train')
            eval_dataset = dataset_class(eval_samples, data_processor, mode='eval')

        model = model_class.from_pretrained(os.path.join(config.model_dir, config.model_name),
                                            num_labels=data_processor.num_labels)

        trainer = trainer_class(config=config, model=model, data_processor=data_processor,
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

        if config.task_name == 'ee':
            test_dataset = dataset_class(test_samples, data_processor, tokenizer, mode='test', ngram_dict=ngram_dict,
                                         max_length=config.max_length, model_type=config.model_type)
        else:
            test_dataset = dataset_class(test_samples, data_processor, mode='test')
            
        model = model_class.from_pretrained(output_dir, num_labels=data_processor.num_labels)
        trainer = trainer_class(config=config, model=model, data_processor=data_processor,
                                tokenizer=tokenizer, logger=logger, model_class=model_class, ngram_dict=ngram_dict)
        trainer.predict(test_dataset=test_dataset, model=model)


if __name__ == '__main__':
    main()
