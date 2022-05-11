
import re
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from transformers import BertTokenizer, BertForMaskedLM
import random
from string import ascii_letters
from datasets import Dataset
from pytorch_lightning.loggers import TensorBoardLogger


class TextDataset(Dataset):
    def __init__(self, text_file, tokenizer):
        target_text = self.load_data(text_file)
        input_text = self.generate_data(target_text)

        self.target_data = tokenizer(target_text, return_tensors="pt")
        self.input_data = tokenizer(input_text, return_tensors="pt")

        del target_text
        del input_text

    def load_data(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()
        return lines

    def generate_data(self, data):
        # very simple replace augmentation
        # todo: add more realistic spelling and grammar errors
        train_data = []
        char_replacement_percentage = 0.01
        for line in data:
            replacement_count = int(len(line)*char_replacement_percentage)
            chars_to_replace = random.sample(
                range(len(line)), replacement_count)
            new_chars = random.choices(ascii_letters, k=replacement_count)

            modifyed_line = list(line)
            for i, char_index in enumerate(chars_to_replace):
                modifyed_line[char_index] = new_chars[i]
            train_data.append("".join(modifyed_line))
        return train_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data = self.input_data[idx]
        target_data = self.data[idx]
        return input_data, target_data


class SpellingBert(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.model.train()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        logits = self.model(**x).logits
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, steps_per_epoch=int(10000/8), epochs=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, train_batch, batch_idx):
        y = train_batch.pop("target_ids")
        target_attention_mask = train_batch.pop("target_attention_mask")
        x = train_batch
        #x = x.view(x.size(0), -1)
        #z = self.encoder(x)
        #logits = self.model(input_ids=torch.stack(x["input_ids"]),token_type_ids=torch.stack(x["token_type_ids"]),attention_mask=torch.stack(x["attention_mask"])).logits
        logits = self.model(**x).logits
        #loss = self.loss(logits.view(-1,self.tokenizer.vocab_size),y.view(-1))
        
        #masked loss
        active_loss = target_attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.tokenizer.vocab_size)
        active_labels = torch.where(
                    active_loss, y.view(-1), torch.tensor(self.loss.ignore_index).type_as(y)
                )
        loss = self.loss(active_logits, active_labels)
        
        self.log('train_loss', loss)
        sch = self.lr_schedulers()
        sch.step()
        self.log("learningrate",sch.get_last_lr()[0])
        return loss

    def validation_step(self, val_batch, batch_idx):
        self.training_step(val_batch, batch_idx)

# data


def replace_augment(text, char_replacement_percentage=0.03):
    replacement_count = int(len(text)*char_replacement_percentage)
    chars_to_replace = random.sample(range(len(text)), replacement_count)
    new_chars = random.choices(ascii_letters, k=replacement_count)

    modifyed_line = list(text)
    for i, char_index in enumerate(chars_to_replace):
        modifyed_line[char_index] = new_chars[i]
    return "".join(modifyed_line).lower()


def map_text(items):
    result = tokenizer(list(
        map(replace_augment, items["input"])), truncation=True, padding="max_length")
        
    target = tokenizer(
        items['input'], truncation=True, padding="max_length")
    result['target_ids'] = target["input_ids"]
    result['target_attention_mask'] = target["attention_mask"]

    return result


def load_data(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    return lines

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    texts = load_data("en.txt")
    texts = [text.lower() for text in texts]
    my_dict = {"input":texts}
    dataset = Dataset.from_dict(my_dict)
    dataset = dataset.map(map_text, batched=True,num_proc=4)
    dataset = dataset.remove_columns("input")
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'target_ids', 'target_attention_mask'])
    dataset = dataset.train_test_split(test_size=0.2)

    train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=8, num_workers=8)
    test_dataloader = DataLoader(dataset["test"], batch_size=16)
    #dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    #mnist_train, mnist_val = random_split(dataset, [55000, 5000])
    #
    #train_loader = DataLoader(mnist_train, batch_size=32)
    #val_loader = DataLoader(mnist_val, batch_size=32)

    # model
    model = SpellingBert()

    #logger

    logger = TensorBoardLogger("tb_logs", name="my_model")
    # training
    trainer = pl.Trainer(logger=logger,gpus=1, precision=16,max_epochs=2)# , limit_train_batches=0.5
    trainer.fit(model, train_dataloader, test_dataloader)
