import random 
from string import ascii_letters
import re
import os

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

def combine_sentences(text, sentences, augmentation_probability = 0.5):
    if random.random() < augmentation_probability and len(text) < 150:
        sentences_to_sample = random.randint(1,3)
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
    with open("data/data.txt",'r') as file:
        sentences = file.readlines(100000)
        sentences = [cleanup(sentence) for sentence in sentences if len(sentence) < 60]
    
    with open("en.csv","w",encoding='utf-8') as output:        
        with open("data/data.txt",'r') as file:
            for line in file:
                line = cleanup(line)
                line = combine_sentences(line,sentences)
                new_line = delete_word(line)
                new_line = delete_characters(new_line)
                new_line = replace_augment(new_line)
                new_line = new_line.lower() # train to reconstruct capitalization
                new_line = new_line.replace(",","").replace(".","").replace("!","").replace("?","")
                output.write(f'"{new_line.strip()}","{line.strip()}"\n')        
    os.system("echo \"text,summary\" > en.train.csv")
    os.system("head -n 100000 en.csv >> en.train.csv")
    os.system("echo \"text,summary\" > en.test.csv")
    os.system("tail -n 10000 en.csv >> en.test.csv")