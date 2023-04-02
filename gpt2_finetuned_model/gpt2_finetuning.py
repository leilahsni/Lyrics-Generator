''' adapted from https://github.com/falloutdurham/beginners-pytorch-deep-learning/blob/master/chapter9/Chapter9.5.ipynb '''

import os
import re
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import torch
from tqdm import tqdm
from tqdm import trange 
import torch.nn.functional as F

class SongLyrics(Dataset):
    ''' dataset object '''

    def __init__(self, dataset, truncate=False, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.max_length = max_length
        self.lyrics = []


        for row in dataset['lyrics']: # creating dataset
          self.lyrics.append(torch.tensor(
                self.tokenizer.encode(f"<|beginoftext|>{str(row)[:self.max_length]}<|endoftext|>") # formating dataset with bos & eos tokens
            ))

        if truncate:
            self.lyrics = self.lyrics[:20000]

        self.lyrics_count = len(self.lyrics)
        
    def __len__(self):
        return self.lyrics_count

    def __getitem__(self, item):
        return self.lyrics[item]

class GPT2LyricsGenerator():
  ''' generator object '''

  def __init__(self, dataset, path='./gpt2_finetuned_model/finetuned_models/gpt2-finetuned-model.h5', max_seq_len=400):

    self.path = path
    self.max_seq_len = max_seq_len

    self.dataset = dataset

    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.model = GPT2LMHeadModel.from_pretrained('gpt2')

  def pack_tensor(self, new_tensor, packed_tensor):
    ''' memory optimization from https://github.com/falloutdurham/beginners-pytorch-deep-learning/blob/master/chapter9/Chapter9.5.ipynb '''

    if packed_tensor is None:
      return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > self.max_seq_len:
      return packed_tensor, False, new_tensor
    else:
      packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
      return packed_tensor, True, None

  def train(self, batch_size=16, epochs=5, warmup_steps=5000, lr=3e-5,
    gpt2_type="gpt2", output_dir="./gpt2_finetuned_model/finetuned_models", output_prefix="lyrics",
    save_model_on_epoch=False,
  ):
    ''' train function adapted from https://github.com/falloutdurham/beginners-pytorch-deep-learning/blob/master/chapter9/Chapter9.5.ipynb '''

    acc_steps = 100
    device = torch.device("cpu")
    self.model.train()

    optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
    loss = 0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        print(loss)
        
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = self.pack_tensor(entry, input_tensor)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = self.model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                self.model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None

        if save_model_on_epoch:
            torch.save(
                self.model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )

    torch.save(self.model, self.path)

    return self.model

  def generate(self, model, prompt, entry_count=10, entry_length=100, top_p=0.8, temperature=1.):
    ''' generate function from https://github.com/falloutdurham/beginners-pytorch-deep-learning/blob/master/chapter9/Chapter9.5.ipynb '''

    model.eval()
    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad(): # no gradient update

      for entry_idx in trange(entry_count):
        entry_finished = False # to control ending of generation
        generated = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)

        for i in range(entry_length):
          outputs = model(generated, labels=generated)
          loss, logits = outputs[:2]
          logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

          sorted_logits, sorted_indices = torch.sort(logits, descending=True)
          cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

          sorted_indices_to_remove = cumulative_probs > top_p # top-p sampling
          sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
          ].clone()
          sorted_indices_to_remove[..., 0] = 0

          indices_to_remove = sorted_indices[sorted_indices_to_remove]
          logits[:, indices_to_remove] = filter_value

          next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
          generated = torch.cat((generated, next_token), dim=1)

          if next_token in self.tokenizer.encode("<|endoftext|>"):
            entry_finished = True # if token generated is <|endoftext|>, generation is finished

          if entry_finished:

            generated_num = generated_num + 1

            output_list = list(generated.squeeze().numpy())
            output_text = self.tokenizer.decode(output_list)
            generated_list.append(output_text)
            break
                
          if not entry_finished:
            output_list = list(generated.squeeze().numpy())
            output_text = f"{self.tokenizer.decode(output_list)}<|endoftext|>" 

            generated_list.append(output_text)
                  
    return generated_list

  def text_generation(self, model, prompt):

    return self.generate(model, prompt, entry_count=1)