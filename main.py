import os.path
from argparse import ArgumentParser
from transformers import Trainer, TrainingArguments

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from model import get_gpt2_model, get_dbert_model
from transformers import AdamW
import torch

def parse():
    parser = ArgumentParser()
    parser.add_argument("train_dir", default=r'C:\studies\datasets\abc_dataset',
                        type=str, help="Path dataset directory")
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--save_steps", default=10, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--n_workers", default=4, type=int)
    parser.add_argument("--min_sequence_lenght", default=16, type=int)
    parser.add_argument("--max_sequence_lenght", default=512, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--output_dir", default=r"models\GptJ", type=str)
    parser.add_argument('--check', action='store_true')
    parser.add_argument("--tokenizer", default=r"tokenizer", type=str)
    parser.add_argument("--contex_len", default=512, type=int)
    parser.add_argument("--model_name", default='gpt2', type=str)
    return parser.parse_args()


def main(args):
    if args.model_name == 'gpt2':
        tokenizer, model = get_gpt2_model(os.path.join("tokenizers", 'ByteLevelBPETokenizer'), args.contex_len)
    else:
        tokenizer, model = get_dbert_model(os.path.join("tokenizers", 'BertWordPieceTokenizer'), args.contex_len)
    data = load_dataset("text", data_files={'train': os.path.join(args.train_dir, "train_yandex.txt"),
                                            'test': os.path.join(args.train_dir, "test_yandex.txt")})

    def encode(lines):
        return tokenizer(lines["text"], truncation=True, add_special_tokens=True, max_length=args.contex_len)

    data.set_transform(encode)
    train_dataset = data["train"]
    valid_dataset = data["test"]
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_args = TrainingArguments(
        output_dir=f"P{args.model_name}_{args.contex_len}",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        prediction_loss_only=True,
        remove_unused_columns=False,
        fp16=True,
        optim="adamw_torch",
        logging_dir=f"logs_{args.model_name}_{args.contex_len}",
        logging_steps=250,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    trainer.train()
    os.makedirs(f"P{args.model_name}_{args.contex_len}", exist_ok=True)
    trainer.save_model(f'MusicTransformer_{args.contex_len}')
    print("Finished")


if __name__ == "__main__":
    args = parse()
    main(args)

