import optuna
from transformers import TrainingArguments, Trainer, BertForQuestionAnswering, BertTokenizerFast
from datasets import load_dataset, load_metric


def prepare_features(examples, tokenizer, max_length=384):
    # Tokenize the inputs with truncation and padding to the maximum length
    tokenized_examples = tokenizer(
        examples['question'], 
        examples['context'],
        truncation="only_second", 
        max_length=max_length,
        stride=128, 
        return_overflowing_tokens=True,
        return_offsets_mapping=True, 
        padding="max_length"
    )
    
    # Map the new tokenized examples to the original example indices
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = [examples["id"][i] for i in sample_mapping]
    
    # Add labels
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        
        # Define the start and end positions
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1
        answers = examples["answers"][sample_mapping[i]]
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        
        token_start_index = 0
        while sequence_ids[token_start_index] != context_index:
            token_start_index += 1
        
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != context_index:
            token_end_index -= 1
        
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)

    tokenized_examples.pop("offset_mapping")
    
    return tokenized_examples

def compute_metrics(p):
    metric = load_metric("squad")
    start_logits = p.predictions[0]
    end_logits = p.predictions[1]
    predictions = (start_logits, end_logits)
    return metric.compute(predictions=predictions, references=p.label_ids)

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-5)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32])
    num_train_epochs = trial.suggest_int('num_train_epochs', 3, 5)

    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)

    dataset = load_dataset('squad')

    # Apply the prepare_features function with padding and truncation
    train_dataset = dataset['train'].map(lambda x: prepare_features(x, tokenizer), batched=True, remove_columns=dataset['train'].column_names)
    validation_dataset = dataset['validation'].map(lambda x: prepare_features(x, tokenizer), batched=True, remove_columns=dataset['validation'].column_names)

    # Verify lengths consistency
    def verify_lengths(dataset):
        for example in dataset:
            input_len = len(example['input_ids'])
            assert input_len == max_length, f"input_ids length {input_len} != {max_length}"
            att_mask_len = len(example['attention_mask'])
            assert att_mask_len == max_length, f"attention_mask length {att_mask_len} != {max_length}"
            if 'token_type_ids' in example:
                token_type_len = len(example['token_type_ids'])
                assert token_type_len == max_length, f"token_type_ids length {token_type_len} != {max_length}"

    max_length = 384
    verify_lengths(train_dataset)
    verify_lengths(validation_dataset)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=16,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result['eval_f1']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

best_trial = study.best_trial
print(f'Best F1 Score: {best_trial.value}')
print(f'Best hyperparameters: {best_trial.params}')
