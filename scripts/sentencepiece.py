import sentencepiece as spm
from constants import *

# Train SentencePiece model
# source model
spm.SentencePieceTrainer.train(
    input=raw_src_file,
    pad_id=0,
    unk_id=3,
    bos_id=1,
    eos_id=2,
    pad_piece='[PAD]',
    unk_piece='[UNK]',
    bos_piece='[BOS]',
    eos_piece='[EOS]',
    model_prefix="src_sp",
    vocab_size=vocab_size,
    model_type=model_type
)

# target model
spm.SentencePieceTrainer.train(
    input=raw_trg_file,
    pad_id=0,
    unk_id=3,
    bos_id=1,
    eos_id=2,
    pad_piece='[PAD]',
    unk_piece='[UNK]',
    bos_piece='[BOS]',
    eos_piece='[EOS]',
    model_prefix="trg_sp",
    vocab_size=vocab_size,
    model_type=model_type
)