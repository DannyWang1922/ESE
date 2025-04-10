import mteb
import sys
import os
import logging

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

import torch
import fcntl
import time
import argparse
from prettytable import PrettyTable
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel
from espresso import Pooler
import numpy as np
    
# Import SentEval
sys.path.insert(0, './SentEval')
import senteval # type: ignore


device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
model = mteb.get_model("BAAI/bge-base-en-v1.5")

# specify what you want to evaluate it on
tasks = ['STS12']
tasks = mteb.get_tasks(tasks=tasks)

# run the evaluation
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model)