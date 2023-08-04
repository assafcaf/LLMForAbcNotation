from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import DistilBertTokenizer, DistilBertForMaskedLM, DistilBertConfig
import torch

def get_gpt2_model(tokenizer_pth, max_len=512):
    print("Loading /{} ... ".format(tokenizer_pth))
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_pth)
    tokenizer.add_special_tokens({"eos_token": "</s>",
                                  "bos_token": "<s>",
                                  "unk_token": "<unk>",
                                  "pad_token": "<pad>",
                                  "mask_token": "<mask>"})
    print("Loading model...")
    config = GPT2Config(vocab_size=len(tokenizer),
                        bos_token=tokenizer.bos_token,
                        eos_token=tokenizer.eos_token,
                        max_position_embeddings=max_len)
    model = GPT2LMHeadModel(config=config)
    return tokenizer, model

def get_dbert_model(tokenizer_pth, max_len=512):
    print("Loading /{} ... ".format(tokenizer_pth))
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_pth, bos_token="<s>", eos_token="</s>", pad_token="<pad>")
    tokenizer.add_special_tokens({"eos_token": "</s>",
                                  "bos_token": "<s>",
                                  "unk_token": "<unk>",
                                  "pad_token": "<pad>",
                                  "mask_token": "<mask>"})
    print("Loading model...")
    config = DistilBertConfig(vocab_size=len(tokenizer),
                              bos_token=tokenizer.bos_token,
                              eos_token=tokenizer.eos_token,
                              max_position_embeddings=max_len)
    model = DistilBertForMaskedLM(config=config)
    return tokenizer, model


