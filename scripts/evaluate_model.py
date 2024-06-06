from transformers import BertForQuestionAnswering, BertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset

def prepare_features(examples, tokenizer):
    tokenized_examples = tokenizer(
        examples['question'], examples['context'],
        truncation="only_second", max_length=384,
        stride=128, return_overflowing_tokens=True,
        return_offsets_mapping=True, padding="max_length"
    )
    return tokenized_examples

def evaluate_model():
    model_name = './models/bert-qa'
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)

    dataset = load_dataset('squad')
    validation_dataset = dataset['validation'].map(lambda x: prepare_features(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )

    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    evaluate_model()

## final Changes