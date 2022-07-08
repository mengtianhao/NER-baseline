import os
import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from cblue.utils import ProgressBar
from cblue.metrics import ee_metric
from cblue.metrics import ee_commit_prediction
from cblue.models import save_zen_model


class Trainer(object):
    def __init__(
            self,
            config,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):

        self.config = config
        self.model = model
        self.data_processor = data_processor
        self.tokenizer = tokenizer

        if train_dataset is not None and isinstance(train_dataset, Dataset):
            self.train_dataset = train_dataset

        if eval_dataset is not None and isinstance(eval_dataset, Dataset):
            self.eval_dataset = eval_dataset

        self.logger = logger
        self.model_class = model_class
        self.ngram_dict = ngram_dict

    def train(self):
        config = self.config
        logger = self.logger
        model = self.model
        model.to(config.device)

        train_dataloader = self.get_train_dataloader()

        num_training_steps = len(train_dataloader) * config.epochs
        num_warmup_steps = num_training_steps * config.warmup_proportion
        num_examples = len(train_dataloader.dataset)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)

        logger.info("***** Running training *****")
        logger.info("Num samples %d", num_examples)
        logger.info("Num epochs %d", config.epochs)
        logger.info("Num training steps %d", num_training_steps)
        logger.info("Num warmup steps %d", num_warmup_steps)

        global_step = 0
        best_step = None
        best_score = .0
        cnt_patience = 0
        for i in range(config.epochs):
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
            for step, item in enumerate(train_dataloader):
                loss = self.training_step(model, item)
                pbar(step, {'loss': loss.item()})

                if config.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad()

                global_step += 1

                if config.logging_steps > 0 and global_step % config.logging_steps == 0:
                    print("")
                    score = self.evaluate(model)
                    if score > best_score:
                        best_score = score
                        best_step = global_step
                        cnt_patience = 0
                        self._save_checkpoint(model, global_step)
                    else:
                        cnt_patience += 1
                        self.logger.info("Earlystopper counter: %s out of %s", cnt_patience, config.earlystop_patience)
                        if cnt_patience >= self.config.earlystop_patience:
                            break
            if cnt_patience >= config.earlystop_patience:
                break

        logger.info("Training Stop! The best step %s: %s", best_step, best_score)
        if config.device == 'cuda':
            torch.cuda.empty_cache()

        self._save_best_checkpoint(best_step=best_step)

        return global_step, best_step

    def evaluate(self, model):
        raise NotImplementedError

    def _save_checkpoint(self, model, step):
        raise NotImplementedError

    def _save_best_checkpoint(self, best_step):
        raise NotImplementedError

    def training_step(self, model, item):
        raise NotImplementedError

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True
        )

    def get_eval_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False
        )

    def get_test_dataloader(self, test_dataset, batch_size=None):
        if not batch_size:
            batch_size = self.config.eval_batch_size

        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )


class EETrainer(Trainer):
    def __init__(
            self,
            config,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(EETrainer, self).__init__(
            config=config,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict
        )

    def training_step(self, model, item):
        model.train()

        input_ids = item[0].to(self.config.device)
        token_type_ids = item[1].to(self.config.device)
        attention_mask = item[2].to(self.config.device)
        labels = item[3].to(self.config.device)

        if self.config.model_type == 'zen':
            input_ngram_ids = item[4].to(self.config.device)
            ngram_attention_mask = item[5].to(self.config.device)
            ngram_token_type_ids = item[6].to(self.config.device)
            ngram_position_matrix = item[7].to(self.config.device)

        if self.config.model_type == 'zen':
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=labels, ngram_ids=input_ngram_ids, ngram_positions=ngram_position_matrix,
                            ngram_attention_mask=ngram_attention_mask, ngram_token_type_ids=ngram_token_type_ids)
        else:
            outputs = model(labels=labels.to(torch.int64), input_ids=input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        loss = outputs[0]
        loss.backward()

        return loss.detach()

    def evaluate(self, model):
        config = self.config
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        eval_labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        for step, item in enumerate(eval_dataloader):
            model.eval()

            input_ids = item[0].to(self.config.device)
            token_type_ids = item[1].to(self.config.device)
            attention_mask = item[2].to(self.config.device)
            labels = item[3].to(self.config.device)

            if config.model_type == 'zen':
                input_ngram_ids = item[4].to(self.config.device)
                ngram_attention_mask = item[5].to(self.config.device)
                ngram_token_type_ids = item[6].to(self.config.device)
                ngram_position_matrix = item[7].to(self.config.device)

            with torch.no_grad():
                if self.config.model_type == 'zen':
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                    labels=labels, ngram_ids=input_ngram_ids,
                                    ngram_positions=ngram_position_matrix,
                                    ngram_token_type_ids=ngram_token_type_ids,
                                    ngram_attention_mask=ngram_attention_mask)
                else:
                    outputs = model(labels=labels, input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)

                # outputs = model(labels=labels, **inputs)
                loss, logits = outputs[:2]
                # active_index = inputs['attention_mask'].view(-1) == 1
                active_index = attention_mask.view(-1) == 1
                active_labels = labels.view(-1)[active_index]
                logits = logits.argmax(dim=-1)
                active_logits = logits.view(-1)[active_index]

            if preds is None:
                preds = active_logits.detach().cpu().numpy()
                eval_labels = active_labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, active_logits.detach().cpu().numpy(), axis=0)
                eval_labels = np.append(eval_labels, active_labels.detach().cpu().numpy(), axis=0)

        p, r, f1, _ = ee_metric(preds, eval_labels)
        logger.info("%s-%s precision: %s - recall: %s - f1 score: %s", config.task_name, config.model_name, p, r, f1)
        return f1

    def predict(self, model, test_dataset):
        config = self.config
        logger = self.logger
        test_dataloader = self.get_test_dataloader(test_dataset)
        num_examples = len(test_dataloader.dataset)
        model.to(config.device)

        predictions = []

        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Prediction')
        for step, item in enumerate(test_dataloader):
            model.eval()

            input_ids = item[0].to(self.onfig.device)
            token_type_ids = item[1].to(self.config.device)
            attention_mask = item[2].to(self.config.device)

            if config.model_type == 'zen':
                input_ngram_ids = item[3].to(self.config.device)
                ngram_attention_mask = item[4].to(self.config.device)
                ngram_token_type_ids = item[5].to(self.config.device)
                ngram_position_matrix = item[6].to(self.config.device)

            with torch.no_grad():
                if self.config.model_type == 'zen':
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                    ngram_ids=input_ngram_ids,
                                    ngram_positions=ngram_position_matrix,
                                    ngram_token_type_ids=ngram_token_type_ids,
                                    ngram_attention_mask=ngram_attention_mask)
                else:
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)

                if config.model_type == 'zen':
                    logits = outputs.detach()
                else:
                    logits = outputs[0].detach()
                # active_index = (inputs['attention_mask'] == 1).cpu()
                active_index = attention_mask == 1
                preds = logits.argmax(dim=-1).cpu()

                for i in range(len(active_index)):
                    predictions.append(preds[i][active_index[i]].tolist())
            pbar(step=step, info="")

        # test_inputs = [list(text) for text in test_dataset.texts]
        test_inputs = test_dataset.texts
        predictions = [pred[1:-1] for pred in predictions]
        predicts = self.data_processor.extract_result(predictions, test_inputs)
        ee_commit_prediction(dataset=test_dataset, preds=predicts, output_dir=config.result_output_dir)

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.config.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.config.model_type == 'zen':
            save_zen_model(output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.config)
        else:
            model.save_pretrained(output_dir)
            torch.save(self.config, os.path.join(output_dir, 'training_config.bin'))
            self.tokenizer.save_vocabulary(save_directory=output_dir)
        self.logger.info('Saving models checkpoint to %s', output_dir)

    def _save_best_checkpoint(self, best_step):
        model = self.model_class.from_pretrained(os.path.join(self.config.output_dir, f'checkpoint-{best_step}'),
                                                 num_labels=self.data_processor.num_labels)
        if self.config.model_type == 'zen':
            save_zen_model(self.config.output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.config)
        else:
            model.save_pretrained(self.config.output_dir)
            torch.save(self.config, os.path.join(self.config.output_dir, 'training_config.bin'))
            self.tokenizer.save_vocabulary(save_directory=self.config.output_dir)
        self.logger.info('Saving models checkpoint to %s', self.config.output_dir)