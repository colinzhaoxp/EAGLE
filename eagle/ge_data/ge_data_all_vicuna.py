import os
import argparse
import copy
from loguru import logger

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from datasets import load_dataset
import json
from fastchat.model.model_adapter import get_conversation_template


parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--model-path', type=str, default='lmsys/vicuna-7b-v1.3')
parser.add_argument("--data-path", type=str, default='ShareGPT_V4.3_unfiltered_cleaned_split.json')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=-1)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]


def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length


def build_dataset_rank(
        tokenizer,
        data_path="ShareGPT_V4.3_unfiltered_cleaned_split.json",
        split="train",
        select=None,
):
    ds = load_dataset('json', data_files=data_path)
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    if args.end == -1:
        args.end = len(ds)
    logger.info(f"ds length: {len(ds)}, select from {args.start} to {args.end}")
    ds1 = ds.select(range(args.start, args.end))
    original_columns1 = ds1.column_names

    num_proc = os.cpu_count() - 2

    def preprocess_function(examples):
        new_examples = {
            "conversation":[],
            "input_ids": [],
            "loss_mask": []
        }
        for i in range(len(examples['id'])):
            conv = get_conversation_template("vicuna")
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            source= examples['conversations'][i]
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversation=conv.get_prompt()

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids[0]
            loss_mask=torch.ones_like(input_ids)

            sep = conv.sep + conv.roles[1] + ": "

            total_len = int(input_ids.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2)
            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    instruction_len -= 1

                # Ignore the user instructions
                loss_mask[cur_len: cur_len + instruction_len] = 0
                cur_len += turn_len

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    cur_len -= 1

            loss_mask[cur_len:] = 0

            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None,:])
            new_examples["loss_mask"].append(loss_mask[None,:])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )


    ds1.set_format(type="torch")

    return ds1


bigtokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
ds = build_dataset_rank(bigtokenizer, args.data_path, "train")
logger.info(ds)

args.outdir = '{}/sharegpt_{}_{}_mufp16'.format(args.outdir, args.start, args.end)
logger.info(f"outdir: {args.outdir}")

bigmodel = AutoModelForCausalLM.from_pretrained(args.model_path,  device_map="auto",torch_dtype=torch.float16)
bigmodel.eval()


@torch.no_grad()
def ge(data):
    input_ids=data["input_ids"]
    outs_big = bigmodel(input_ids.cuda(), output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1]
    # max_prob_tokens_big = torch.argmax(outs_big.logits, dim=-1)
    # probs = torch.softmax(outs_big.logits, dim=-1)
    # maxp=probs[0].max(dim=1).values
    td={"input_ids":input_ids.cpu()[0],"hidden_state":hidden_state_big.cpu()[0],"loss_mask":data["loss_mask"].cpu()[0]}
    return td


outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)


def writedata(name, data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')


for data in tqdm(ds):
    outdata = ge(data)
    writedata(outdir,outdata)