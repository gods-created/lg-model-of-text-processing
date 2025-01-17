from enum import Enum
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
)
from typing import Optional, List, Any
from datasets import Dataset
import pandas as pd
from collections import namedtuple
from os.path import exists
from loguru import logger

MODEL_SOURCE_DIR = './model'
MODEL_SOURCE = 'google-t5/t5-base'

class Targets(Enum):
    TRAINING = 'training'
    JOB = 'job'

def namedtuple_generator(name: str, keys: List[str], values: List[Any]) -> namedtuple:
    TupleClass = namedtuple(name, keys)
    return TupleClass(*values)

def model_and_tokenizer() -> namedtuple:
    source = MODEL_SOURCE_DIR if exists(MODEL_SOURCE_DIR) else MODEL_SOURCE

    model = T5ForConditionalGeneration.from_pretrained(source).to('cpu')
    tokenizer = T5Tokenizer.from_pretrained(source, legacy=False)

    return namedtuple_generator(
        'model_and_tokenizer', 
        ['model', 'tokenizer'], 
        [model, tokenizer]
    )

def file_content(filename: str) -> list:
    table = pd.read_csv(filename)
    data = table.iloc[:].fillna(0).values.tolist()
    return data

def dataset(filename: str, tokenizer: T5Tokenizer) -> Optional[Dataset]:
    data = file_content(filename)
    if not data:
        logger.error('The file is empty')
        return None
    
    prepared_data = [
        {
            'input_id': tokenizer(
                ' '.join(item[:-1]),
                padding=True,
                return_tensors='pt',
                truncation=False
            ).input_ids.squeeze(),
            'label': tokenizer(
                item[-1],
                padding=True,
                return_tensors='pt',
                truncation=False
            ).input_ids.squeeze(),
        }

        for item in data if all(item)
    ]

    return Dataset.from_dict({
        'input_ids': [item['input_id'] for item in prepared_data],
        'labels': [item['label'] for item in prepared_data]
    })

def training(filename: str) -> str:
    try:
        model_and_tokenizer_response = model_and_tokenizer()
        model, tokenizer = (
            model_and_tokenizer_response.model,
            model_and_tokenizer_response.tokenizer
        )

        dataset_response = dataset(filename, tokenizer)
        if not dataset_response:
            return 'The training didn\'t end right'

        training_args = TrainingArguments(
            output_dir='./settings',
            num_train_epochs=3,
            per_device_train_batch_size=4
        ) 

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
        )

        train = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset_response
        )

        train.train()

        model.save_pretrained(MODEL_SOURCE_DIR)
        tokenizer.save_pretrained(MODEL_SOURCE_DIR)

        return 'The training finished success'

    except (Exception, ) as e:
        return str(e)
    
def job(filename: str) -> str:
    try:
        if not exists(MODEL_SOURCE_DIR):
            raise FileExistsError(
                'The model not created'
            )
            
        model_and_tokenizer_response = model_and_tokenizer()
        model, tokenizer = (
            model_and_tokenizer_response.model,
            model_and_tokenizer_response.tokenizer
        )

        data = file_content(filename)
        if not data:
            return 'The file is empty'
        
        responses = []
        for item in data:
            sentence = ' '.join(item)
            tokenized_data = tokenizer(
                sentence,
                padding=True,
                return_tensors='pt',
                truncation=False
            ).to(model.device)
            output_data, *_ = model.generate(
                tokenized_data['input_ids'],
                max_new_tokens=5000
            )

            response = tokenizer.decode(output_data, skip_special_tokens=True)
            responses.append(response)

        new_filename = f'with_responses__{filename}'

        table = pd.read_csv(filename)
        table['RESPONSES'] = responses
        table.to_csv(new_filename, index=False)

        return f'Obtaining the predictions ended successfully, checking \'{new_filename}\' file'

    except (FileExistsError, Exception, ) as e:
        return str(e)