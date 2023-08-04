import os
import time
import random
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from utils import parse_notation

FROM_LOCAL = True
out_dir = Path(r'C:\studies\datasets\abc_dataset')
out_dir.mkdir(exist_ok=True)
sep = "<sep>"
br = "<nl>"
if FROM_LOCAL:
    samples = []
    for dir_name, _, files in os.walk(r"C:\studies\datasets\yandex"):
        for file in tqdm(files, desc="loading data", total=len(files)):
            f_name = os.path.join(dir_name, file)
            if f_name.endswith(".abc"):
                samples.append(f_name)
else:
    samples = load_dataset("sander-wood/massive_abcnotation_dataset")["train"]
print("preprocessing data...")
time.sleep(0.01)

for sample_idx, sample in tqdm(enumerate(samples), total=len(samples), desc="preprocessing data"):
    if FROM_LOCAL:
        sample_ = {}
        with open(sample, "r") as f:
            sample = {"abc notation": f.read(), "control code": ""}
    keys, notes, control_code = parse_notation(sample)
    line = (control_code + sep + keys + sep + notes).replace("\n", br) + "\n"
    if random.random() < 0.99:
        with open(out_dir / f"train{'_yandex' if FROM_LOCAL else ''}.txt", "a") as f:
            f.write(line)
    else:
        with open(out_dir / f"test{'_yandex' if FROM_LOCAL else ''}.txt", "a") as f:
            f.write(line)

