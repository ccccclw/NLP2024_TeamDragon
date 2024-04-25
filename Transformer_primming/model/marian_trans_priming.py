import os
import logging
import json
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import random_split
from transformers import AdamW, get_scheduler, AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu.metrics import BLEU
import sys
sys.path.append('../../')
from tools import set_random_seed
from argument_parser import get_arguments
from data_loader import TranslationDataset, configure_data_loader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(args, loader, model, optimizer, scheduler, epoch_index, cumulative_loss):
    model.train()
    progress_bar = tqdm(loader, desc=f'Epoch {epoch_index + 1}')
    for step, batch in enumerate(loader, start=1):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        output = model(**batch)
        loss = output.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        cumulative_loss += loss.item()
        progress_bar.set_description(f'Average Loss: {cumulative_loss / (len(loader) * (epoch_index + 1) + step):.4f}')

    return cumulative_loss

def evaluate_model(args, loader, model, tokenizer):
    model.eval()
    predictions, references = [], []
    bleu_metric = BLEU()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            generated_tokens = model.generate(**batch).cpu().numpy()
            labels = batch['labels'].cpu().numpy()

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(decoded_preds)
            references.extend([[label] for label in decoded_labels])

    return bleu_metric.corpus_score(predictions, references).score

def execute_training_phase(args, train_loader, dev_loader, model, tokenizer):
    total_steps = len(train_loader) * args.num_epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps)

    best_bleu_score = 0
    total_loss = 0
    for epoch in range(args.num_epochs):
        logger.info(f'Starting Epoch {epoch+1}/{args.num_epochs}')
        total_loss = train_model(args, train_loader, model, optimizer, scheduler, epoch, total_loss)
        
        current_bleu = evaluate_model(args, dev_loader, model, tokenizer)
        logger.info(f'Epoch {epoch+1} - Dev BLEU: {current_bleu:.4f}')
        
        if current_bleu > best_bleu_score:
            best_bleu_score = current_bleu
            model_path = os.path.join(args.output_dir, f'best_model_epoch_{epoch+1}_bleu_{current_bleu:.4f}.bin')
            torch.save(model.state_dict(), model_path)
            logger.info(f'New best model saved to {model_path}')

if __name__ == '__main__':
    args = get_arguments()
    set_random_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint).to(args.device)

    if args.do_train:
        training_data = TranslationDataset(args.train_file)
        train_set, dev_set = random_split(training_data, [TRAIN_SET_SIZE, VALID_SET_SIZE])
        train_loader = configure_data_loader(args, train_set, tokenizer, shuffle=True)
        dev_loader = configure_data_loader(args, dev_set, tokenizer, shuffle=False)
        execute_training_phase(args, train_loader, dev_loader, model, tokenizer)

    if args.do_test:
        testing_data = TranslationDataset(args.test_file)
        test_loader = configure_data_loader(args, testing_data, tokenizer, shuffle=False)
        test_bleu = evaluate_model(args, test_loader, model, tokenizer)
        logger.info(f'Test BLEU: {test_bleu:.4f}')
