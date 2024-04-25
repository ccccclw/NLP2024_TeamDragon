import json
import torch
from torch.utils.data import Dataset, DataLoader

# Constants
MAX_DATASET_SIZE = 220000
TRAIN_SET_SIZE = 200000
VALID_SET_SIZE = 20000

class TranslationDataset(Dataset):
    """Dataset class for loading translation data for training a Transformer model."""
    
    def __init__(self, file_path):
        self.samples = self._load_data(file_path)
    
    def _load_data(self, file_path):
        data = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                if idx >= MAX_DATASET_SIZE:
                    break
                data[idx] = json.loads(line.strip())
        return data
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

def create_data_loader(configuration, dataset, model, tokenizer):
    """Creates a DataLoader with a custom collate function tailored for translation tasks."""
    
    def collate_batch(batch_samples):
        src_texts, tgt_texts = [], []
        for sample in batch_samples:
            src_texts.append(sample['chinese'])
            tgt_texts.append(sample['english'])

        tokenized_inputs = tokenizer(
            src_texts, 
            padding=True, 
            max_length=configuration.max_input_length,
            truncation=True, 
            return_tensors="pt"
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                tgt_texts, 
                padding=True, 
                max_length=configuration.max_target_length,
                truncation=True, 
                return_tensors="pt"
            )["input_ids"]

            # Prepare decoder input ids and labels for training
            decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)
            labels[labels == tokenizer.eos_token_id] = -100  # Mask out tokens after eos

            tokenized_inputs['decoder_input_ids'] = decoder_input_ids
            tokenized_inputs['labels'] = labels

        return tokenized_inputs
    
    return DataLoader(
        dataset, 
        batch_size=configuration.batch_size, 
        shuffle=True, 
        collate_fn=collate_batch
    )
