from Config.Model_Config import Model_config
from baselines.run_classifier import main


if __name__ == '__main__':
    config = Model_config(task_name='ee', model_name="chinese-bert-wwm-ext")
    main(config, do_predict=True)
