from lib2to3.pgen2.tokenize import tokenize
import random 
from string import ascii_letters, punctuation
import re
import os
from transformers import AutoTokenizer
from tqdm import tqdm

def tokenizer_check_if_text_too_long(text, tokenizer):
    data = tokenizer.batch_encode_plus([text],max_length=1024,truncation=True,return_overflowing_tokens=True )    
    if len(data["input_ids"]) > 1:
        return True
    else:
        return False#, len(data["input_ids"][0])

def delete_characters(text, char_delete_percentage=0.02):
    modifyed_line = []   
    for char in text:
        if random.random() > char_delete_percentage:
            modifyed_line.append(char)
    return "".join(modifyed_line)


def replace_augment(text, char_replacement_percentage=0.02):
    replacement_count = int(len(text)*char_replacement_percentage)
    chars_to_replace = random.sample(range(len(text)), replacement_count)
    new_chars = random.choices(ascii_letters, k=replacement_count)

    modifyed_line = list(text)
    for i, char_index in enumerate(chars_to_replace):
        modifyed_line[char_index] = new_chars[i]
    return "".join(modifyed_line)

clean_chars = re.compile(r'[^A-Za-zöäüÖÄÜß,.!?’\'$%€0-9\(\)\- ]', re.MULTILINE)
def cleanup(text):    
    text = clean_chars.sub('', text)
    #text = text.replace("\n", "")
    #text = text.replace('"','\\"')
    return text

clean_punctuation = re.compile(r"(?<!\d)[.,;:'?!](?!\d)")
def remove_punctuation(text):
    """Remove all punctuation from string, except if it's between digits"""
    return clean_punctuation.sub("", text)

def combine_sentences(text, sentences, augmentation_probability = 1):
    if random.random() < augmentation_probability:
        sentences_to_sample = random.randint(1,10)
        augmentation_sentences = random.sample(sentences,sentences_to_sample)    
        return text + " " + " ".join(augmentation_sentences)
    else:
        return text

def delete_word(text, augmentation_probability = 0.05):
    if random.random() < augmentation_probability:
        words = text.split()
        word_to_remove = random.randint(0,len(words)-1)
        words.pop(word_to_remove)
        return " ".join(words)
    else:
        return text


if __name__ == "__main__":
    data_file = "data/data.txt"
    language = "en"
    num_lines = sum(1 for line in open(data_file,'r'))

    with open(data_file,'r') as file:
        sentences = file.readlines(int(num_lines*0.5))
        sentences = [cleanup(sentence) for sentence in sentences]
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    with open(language+".csv","w",encoding='utf-8') as output:        
        with open(data_file,'r') as file:
            for line in tqdm(file, total=num_lines):
                line = cleanup(line)
                line = combine_sentences(line,sentences)                
                if tokenizer_check_if_text_too_long(line,tokenizer):
                    print("text too long ({len(line)}):\n"+line)
                
                new_line = delete_word(line)
                new_line = delete_characters(new_line)
                new_line = replace_augment(new_line)
                new_line = new_line.lower() # train to reconstruct capitalization
                new_line = remove_punctuation(new_line)
                output.write(f'"{new_line.strip()}","{line.strip()}"\n')        
    os.system(f"echo \"text,summary\" > {language}.train.csv")
    os.system(f"head -n 299000 {language}.csv >> {language}.train.csv")
    os.system(f"echo \"text,summary\" > {language}.test.csv")
    os.system(f"tail -n 1000 {language}.csv >> {language}.test.csv")