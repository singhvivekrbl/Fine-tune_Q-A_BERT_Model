from transformers import BertForQuestionAnswering, TrainingArguments, Trainer, BertTokenizerFast
from datasets import load_dataset

def prepare_features(examples, tokenizer):
    tokenized_examples = tokenizer(
        examples['question'], examples['context'],
        truncation='only_second', max_length=384,
        stride=128, return_overflowing_tokens=False,
        return_offsets_mapping=True, padding='max_length'
    )

    return tokenized_examples

def train_model():
    model_name = 'bert-base-uncased'
    tokenizer=BertTokenizerFast.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)

    datasets=load_dataset('squad')
    train_dataset = datasets['train'].map(lambda examples: prepare_features(examples, tokenizer), batched=True)
    validation_dataset = datasets['validation'].map(lambda x: prepare_features(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5, 
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model('models/bert_qa')



if __name__ == "__main__":
    train_model()