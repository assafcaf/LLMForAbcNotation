import torch
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel


print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("tokenizers/ByteLevelBPETokenizer")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.add_special_tokens({"eos_token": "</s>",
                              "bos_token": "<s>",
                              "unk_token": "<unk>",
                              "pad_token": "<pad>",
                              "mask_token": "<mask>"})
print("Loading model...")
model = GPT2LMHeadModel.from_pretrained('C:\studies\ML\MusicGeneration\Pgpt2_512\checkpoint-15000').to('cuda')
with torch.no_grad():
    while True:
        inp = input(">>> ")
        input_ids = tokenizer(inp, return_tensors='pt').to('cuda')
        beam_output = model.generate(
            **input_ids,
            max_length=512,
            num_beams=10,
            no_repeat_ngram_size=15,
            early_stopping=True,
        )
        for beam in beam_output:
            out = tokenizer.decode(beam)
            print(out.replace("<nl>", "\n").replace('<sep>', ""))
            print("------------------------------------------------------------")


