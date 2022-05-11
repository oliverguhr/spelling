
from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForMaskedLM.from_pretrained("bert-base-cased")

inputs = tokenizer("The capitel of France is [MASK].", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

#predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
predicted_token_ids = logits.argmax(axis=-1)
tokenizer.decode(predicted_token_ids[0])
tokenizer.batch_decode(predicted_token_ids)