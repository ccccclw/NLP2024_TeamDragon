# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 21:31:29 2024

@author: ALKAIM
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter
from terminaltables import AsciiTable

import model
from bleu import bleu
from dataset import dataset
from util import convert_data, convert_str, invert_vocab, load_vocab, sort_batch

# Define logger
logger = logging.getLogger(__name__)

# Define configuration
config = {
    "data": {
        "src_vocab": "./data/cn.voc.pkl",
        "trg_vocab": "./data/en.voc.pkl",
        "src_max_len": 50,
        "trg_max_len": 50,
        "train_src": "./data/cn.txt", 
        "train_trg": "./data/en.txt",
        "valid_src": "./data/cn.test.txt",
        "valid_trg": "./data/en.test.txt",
        "eval_script": "scripts/multi-bleu.perl",
    },
    "model": {
        "name": "AttEncDecRNN",
        "checkpoint": "",
        "enc_num_input": 512,
        "dec_num_input": 512,
        "enc_num_hidden": 1024,
        "dec_num_hidden": 1024,
        "dec_natt": 1000,
        "nreadout": 620,
        "enc_emb_dropout": 0.4,
        "dec_emb_dropout": 0.4,
        "enc_hid_dropout": 0.4,
        "readout_dropout": 0.4,
    },
    "training": {
        "optim": "RMSprop",
        "batch_size": 64,
        "lr": 0.0005,
        "l2": 0.0,
        "clip_grad": 1.0,
        "finetuning": False,
        "decay_lr": True, 
        "half_epoch": False,
        "epoch_best": True,
        "restore": False,
        "epoch": 0,
        "nepoch": 40,
        "vfreq": 3,
        "sfreq": 3,
        "beam_size": 10,
    },
    "bookkeeping": {
        "name": None,
        "info": None,
        "seed": 12345,
        "checkpoint": "checkpoint",
        "freq": None,
    },
}

# TensorBoardX writer
writer = SummaryWriter()

# CUDA settings
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Set random seed
torch.manual_seed(config["bookkeeping"]["seed"])
if use_cuda:
    torch.cuda.manual_seed(config["bookkeeping"]["seed"])

# Load vocabulary
src_vocab_stoi = load_vocab(config["data"]["src_vocab"])
src_vocab_itos = invert_vocab(src_vocab_stoi)
trg_vocab_stoi = load_vocab(config["data"]["trg_vocab"])
trg_vocab_itos = invert_vocab(trg_vocab_stoi)

# Special tokens
UNK, PAD, SOS, EOS = "<unk>", "<pad>", "<sos>", "<eos>"

# Update vocabularies in config
config["data"].update({
    "enc_pad": src_vocab_stoi[PAD],
    "dec_sos": trg_vocab_stoi[SOS],
    "dec_eos": trg_vocab_stoi[EOS],
    "dec_pad": trg_vocab_stoi[PAD],
    "enc_num_token": len(src_vocab_stoi),
    "dec_num_token": len(trg_vocab_stoi),
})

# Load parallel dataset
train_dataset = dataset(
    config["data"]["train_src"], config["data"]["train_trg"],
    config["data"]["src_max_len"], config["data"]["trg_max_len"]
)

valid_dataset = dataset(
    config["data"]["valid_src"], config["data"]["valid_trg"]
)

def collate_batch(batch):
    return list(zip(*batch))

train_iter = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    collate_fn=collate_batch
)

valid_iter = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_batch
)

# Save config to a table for logging
config_table = [["key", "value"]]
for k, v in config.items():
    if isinstance(v, dict):
        for kk, vv in v.items():
            config_table.append([f"{k}:{kk}", str(vv)])
    else:
        config_table.append([k, str(v)])

logger.info(f"Config:\n{AsciiTable(config_table).table}")

# Initialize model
model = getattr(model, config["model"]["name"])(config["model"]).to(device)

# Initialize model parameters
for p in model.parameters():
    p.data.uniform_(-0.1, 0.1)
    
# Load checkpoint if provided
if config["model"]["checkpoint"]:  
    model.load_state_dict(torch.load(
        config["model"]["checkpoint"], map_location=device
    ))
    
# Define optimizer   
optimizer = getattr(optim, config["training"]["optim"])(
    model.parameters(), 
    lr=config["training"]["lr"],
    weight_decay=config["training"]["l2"]
)

# Training state
state = {
    "scores": [],
    "epoch_best_score": -float("inf"), 
    "cur_lr": ", ".join([f"{g['lr']:.2e}" for g in optimizer.param_groups]),
    "checkpoint": {
        "best": None,
        "epoch_best": None,
    },
}

def save_model(epoch, score):
    """Save model checkpoint."""
    date = time.strftime("%m-%d|%H:%M", time.localtime())
    filename = (
        f"model-{config['model']['name']}-"
        f"epoch{epoch}_score{score:.4f}-"
        f"{date}.pt"
    )
    file_path = Path(config["bookkeeping"]["checkpoint"]) / filename
    torch.save(model.state_dict(), file_path)
    logger.info(f"Saved model to {file_path}")
    return filename

def load_checkpoint(filename):
    """Load model checkpoint."""
    file_path = Path(config["bookkeeping"]["checkpoint"]) / filename
    model.load_state_dict(torch.load(file_path, map_location=device))
    logger.info(f"Loaded checkpoint from {file_path}")

def train_epoch(epoch):
    """Train model for one epoch."""
    model.train()
    
    for batch_idx, batch in enumerate(train_iter, 1):
        batch = sort_batch(batch)

        src_raw, trg_raw = batch
        src, src_mask = convert_data(
            src_raw, src_vocab_stoi, device, True, UNK, PAD, SOS, EOS
        )
        f_trg, f_trg_mask = convert_data(
            trg_raw, trg_vocab_stoi, device, False, UNK, PAD, SOS, EOS
        )
        b_trg, b_trg_mask = convert_data(
            trg_raw, trg_vocab_stoi, device, True, UNK, PAD, SOS, EOS
        )

        optimizer.zero_grad()
        loss, w_loss = model(src, src_mask, f_trg, f_trg_mask, b_trg, b_trg_mask)

        loss.mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_grad"])
        optimizer.step()

        step = (epoch - 1) * len(train_iter) + batch_idx
        writer.add_scalar("loss", loss.item(), step)

        if batch_idx % 10 == 0 or batch_idx == len(train_iter):
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(train_iter)}] loss: {loss.item():.3f}, "
                f"lr: {state['cur_lr']}"
            )
       
        if batch_idx % config["training"]["vfreq"] == 0:
            score = validate(epoch, step)
            
            if config["training"]["decay_lr"] and len(state["scores"]) > 1:
                if score < 0.99 * state["scores"][-2][0]:                    
                    if config["training"]["restore"]:
                        load_checkpoint(state["checkpoint"]["best"])
                    state["cur_lr"] = ", ".join([
                        f"{g['lr'] * 0.5:.2e}" for g in optimizer.param_groups
                    ])
                    for param_group in optimizer.param_groups:
                        param_group["lr"] *= 0.5
                    logger.info(f"Decay lr to {state['cur_lr']}")
             
            # Save checkpoint       
            if len(state["scores"]) == 1 or score > max(s[0] for s in state["scores"][:-1]):
                if state["checkpoint"]["best"] is not None:
                    Path(config["bookkeeping"]["checkpoint"], state["checkpoint"]["best"]).unlink()
                state["checkpoint"]["best"] = save_model(epoch, score)
                
            if config["training"]["epoch_best"] and score > state["epoch_best_score"]:
                state["epoch_best_score"] = score
                if state["checkpoint"]["epoch_best"] is not None:
                    Path(config["bookkeeping"]["checkpoint"], state["checkpoint"]["epoch_best"]).unlink()
                state["checkpoint"]["epoch_best"] = save_model(epoch, score)
                  
        if batch_idx % config["training"]["sfreq"] == 0:
            sample(epoch, batch_idx, src_raw, trg_raw)

def validate(epoch, step):
    """Validate model."""
    model.eval()
    hyp_list, ref_list = [], []
    
    with torch.no_grad():
        for batch in valid_iter:
            src_raw = batch[0]
            trg_raw = batch[1:]
            src, src_mask = convert_data(
                src_raw, src_vocab_stoi, device, True, UNK, PAD, SOS, EOS
            )
            output = model.beamsearch(src, src_mask, config["training"]["beam_size"], normalize=True)
            best_hyp, _ = output[0]
            best_hyp = convert_str([best_hyp], trg_vocab_itos)
            hyp_list.append(best_hyp[0])
            ref_list.append([x[0] for x in trg_raw])

    hyp_list = [" ".join(x) for x in hyp_list]
    
    # Save hypotheses to temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as hyp_file:
        hyp_file.write("\n".join(hyp_list))
        hyp_path = hyp_file.name
        
    # Calculate multi-bleu score    
    ref_stem = config["data"]["valid_trg"][:-1] + "*"
    cmd = f"cat {hyp_path} | {config['data']['eval_script']} {ref_stem}"
    multi_bleu = float(subprocess.check_output(cmd, shell=True, text=True))
    logger.info(f"Validation multi-bleu: {multi_bleu:.2f}")
    
    # Delete temporary file
    Path(hyp_path).unlink()

    # Calculate bleu score    
    bleu_1 = bleu(hyp_list, ref_list, smoothing=True, n=1)
    bleu_2 = bleu(hyp_list, ref_list, smoothing=True, n=2)
    bleu_3 = bleu(hyp_list, ref_list, smoothing=True, n=3)
    bleu_4 = bleu(hyp_list, ref_list, smoothing=True, n=4)

    for name, score in zip(
        ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "multi-bleu"],
        [bleu_1, bleu_2, bleu_3, bleu_4, multi_bleu]
    ):  
        writer.add_scalar(f"valid/{name}", score, step)
    
    bleu_info = [
        ["Type", "Score"],
        ["bleu-1", f"{bleu_1:.2f}"],
        ["bleu-2", f"{bleu_2:.2f}"],
        ["bleu-3", f"{bleu_3:.2f}"],
        ["bleu-4", f"{bleu_4:.2f}"],
        ["multi-bleu", f"{multi_bleu:.2f}"],
    ]
    
    logger.info(
        f"BLEU scores for validation at step {step}:\n"
        f"{AsciiTable(bleu_info).table}"
    )
    
    state["scores"].append((
        multi_bleu, bleu_1, bleu_2, bleu_3, bleu_4, epoch
    ))

    model.train()
    return multi_bleu

def sample(epoch, batch_idx, src_raw, trg_raw):
    """Perform sampling."""
    model.eval()

    ix = np.random.randint(0, len(src_raw))
    src, src_mask = convert_data(
        [src_raw[ix]], src_vocab_stoi, device, True, UNK, PAD, SOS, EOS
    )
    
    with torch.no_grad():
        output = model.beamsearch(src, src_mask, config["training"]["beam_size"])


    best_hyp, best_score = output[0]
    best_hyp = convert_str([best_hyp], trg_vocab_itos)[0]

    sample_info = [
        ["Type", "Text"],
        ["Source", " ".join(src_raw[ix])], 
        ["Target", " ".join(trg_raw[ix])],
        ["Hypothesis", " ".join(best_hyp)],
    ]

    logger.info(
        f"Sampling at epoch {epoch} batch {batch_idx}:\n"
        f"{AsciiTable(sample_info).table}\n"   
        f"Hypothesis score: {best_score:.2f}\n"
    )

    model.train()


if __name__ == "__main__":
    # Set logger
    log_path = (
        Path("log") / 
        f"log-{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.txt"
    )
    log_path.parent.mkdir(exist_ok=True)

    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        "%(levelname)s - %(message)s"
    ))
    logger.addHandler(stream_handler)

    # Train
    for epoch in range(config["training"]["epoch"], 
                       config["training"]["epoch"] + config["training"]["nepoch"]):
        train_epoch(epoch)

    # Choose the model with the best multi-bleu score    
    best_score_index = max(range(len(state["scores"])), key=lambda i: state["scores"][i][0])
    best_epoch = state["scores"][best_score_index][-1]
    best_score = state["scores"][best_score_index][0]
    logger.info(f"Best model: epoch {best_epoch}, multi-bleu score {best_score:.2f}")
    