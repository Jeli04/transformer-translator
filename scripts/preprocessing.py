import torch
import sentencepiece as spm

def create_markers(input_file_path, output_file_path):
    # Read input file, modify sentences, and write to output file
    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
        open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            english_sentence, spanish_sentence = line.strip().split('\t')  # Assuming sentences are separated by tabs

            # Add language markers
            modified_english = '[EN] ' + english_sentence
            modified_spanish = '[ES] ' + spanish_sentence

            # Write modified sentences to output file
            output_file.write(modified_english + '\t' + modified_spanish + '\n')


# create_markers("data/spa.txt", "data/spa_markers.txt")

"""
    Split Data function 
    splits the EN and ES in a array of pairs (index 0 is en and index 1 is es)
"""
def split_data():
    text_file = "data\spa.txt"

    with open(text_file, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")[:-1]

    text_pairs = []
    for line in lines:
        eng, spa = line.split("\t")
        eng = eng.lower()
        spa = spa.lower()
        text_pairs.append((eng, spa))

    return text_pairs

"""
    Tokenize function
    converts the array of pairs into tokens using sentencepiece
    if the pairs are not equal size pad up the smaller one wiht 0s

    MODIFICATIONS
    use two different tokenizers for each language
    pad the size of the tokens to the block size
    replace the vocab size in the transformer with the size of the tokenizer
"""

def tokenize(data, en_tokenizer, es_tokenizer, block_size):
    # sp = spm.SentencePieceProcessor()
    # sp.Load("models/sentencepiece_model_16k.model")

    tokenized_data = []

    # Get BOS and EOS IDs
    bos = en_tokenizer.bos_id()  
    eos = en_tokenizer.eos_id()
    
    for row in data:
        en = torch.tensor(en_tokenizer.EncodeAsIds(row[0]), dtype=torch.long)
        es = torch.tensor(es_tokenizer.EncodeAsIds(row[1]), dtype=torch.long)

        # Add BOS and EOS
        en = torch.cat([torch.tensor([bos]), en, torch.tensor([eos])]) 
        es = torch.cat([torch.tensor([bos]), es, torch.tensor([eos])])

        len1 = en.size(0)
        len2 = es.size(0)

        # Pad tensors if needed
        if len1 < block_size:
            padding = torch.zeros(block_size - len1, dtype=torch.long)
            en = torch.cat([en, padding])

        if len2 < block_size:
            padding = torch.zeros(block_size - len2, dtype=torch.long)
            es = torch.cat([es, padding])


        tokenized_data.append((en, es))

    return tokenized_data

# en_sp = spm.SentencePieceProcessor()
# en_sp.Load("models/sentencepiece_model_10k_english.model")

# es_sp = spm.SentencePieceProcessor()
# es_sp.Load("models/sentencepiece_model_16k_spanish.model")


# print(tokenize(split_data(), en_sp, es_sp, 256)[0]) # 0 is english
# print(tokenize(split_data(), en_sp, es_sp, 256)[1]) # 1 is spanish
# print(torch.tensor(es_sp.EncodeAsIds("como suele haber varias páginas web sobre cualquier tema, normalmente sólo le doy al botón de retroceso cuando entro en una página web que tiene anuncios en ventanas emergentes. simplemente voy a la siguiente página encontrada por google y espero encontrar algo menos irritante."), dtype=torch.long).size(0))