import json
import wandb
import pandas as pd
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
from transformers import BertConfig, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from custom_datasets import CustomDataset

def train_mlm(wandb_key: str,
              model_name: str,
              mlm_probability: float,
              num_epochs: int,
              training_data_path: str,
              output_model_name: str,
              training_arg_dir: str,
              batch_size: int,
              do_lower_case: bool = True):

    # wandb is used for logging - login with your authentication key
    wandb.login(key=wandb_key)

    # initialise the existing tokeniser and model.
    bert_tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    config = BertConfig()
    model = BertForMaskedLM(config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=bert_tokenizer,
        mlm=True,
        mlm_probability = mlm_probability
    )

    training_args = TrainingArguments(
        output_dir= training_arg_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=2000,
        save_total_limit=2,
        prediction_loss_only=True
    )

    train_dataset = pd.read_csv(training_data_path)
    train_dataset['text'] = train_dataset.sentence

    MAX_LEN = 50
    training_set = CustomDataset(train_dataset, bert_tokenizer, MAX_LEN)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=training_set,
    )
    trainer.train()
    trainer.save_model(output_model_name)

if __name__=='__main__':
    user_settings = json.load(open('user_settings.json'))
    wandb_key = user_settings['wandb_key']
    train_mlm(wandb_key=wandb_key,
              model_name='bert-base-uncased',
              mlm_probability=0.15,
              num_epochs=5,
              training_data_path=r'data/ClinNotes_sentences.csv',
              output_model_name=r"models/bert_mlm_sentences",
              training_arg_dir='training_arguments',
              batch_size=32,
              do_lower_case=True)
