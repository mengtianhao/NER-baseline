from Config.Model_Config import Model_config
from baselines.run_classifier import main


if __name__ == '__main__':
    # config = Model_config(task_name='ee', classify_name='CRF', model_name="chinese-bert-wwm-ext")
    config = Model_config(task_name='ee', classify_name='CRF', model_name="chinese-bert-wwm-ext", mid_struct='bilstm')
    # config = Model_config(task_name='ee', classify_name="BertForTokenClassification", model_name="chinese-bert-wwm-ext")
    main(config, do_predict=True)
