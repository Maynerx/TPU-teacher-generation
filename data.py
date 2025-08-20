from curses import raw
from torch import ne
from torch.utils.data import DataLoader
from datasets import load_dataset, Features, Sequence, Value, Dataset

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES = Features({
    "encoder_inputs": Sequence(Value("int32")),  # [L]
    "decoder_inputs": Sequence(Value("int32")),  # [L]
    "labels":         Sequence(Value("int32")),  # [L]
    "top_k_indices":  Sequence(Sequence(Value("int32"))),    # [L × K]
    "top_k_probs":    Sequence(Sequence(Value("float16")))   # [L × K]
})


# --------- Data Processing Functions --------- #
import random

def random_spans_mask(length, num_noise, mean_span_length=3):
    """Return a boolean mask of which tokens to corrupt."""
    mask = [False] * length
    num_masked = 0
    i = 0
    while num_masked < num_noise and i < length:
        span_len = max(1, int(random.expovariate(1.0 / mean_span_length)))
        if num_masked + span_len > num_noise:
            span_len = num_noise - num_masked
        # mark span
        for j in range(span_len):
            if i + j < length:
                mask[i + j] = True
        num_masked += span_len
        i += span_len + random.randint(1, mean_span_length)
    return mask

def span_corrupt(ids, noise_density=0.15, mean_span_length=3, 
                 bos=None, eos=None, pad=None, sentinel_start=None):
    L = len(ids)
    num_noise = max(1, int(L * noise_density))
    mask = random_spans_mask(L, num_noise, mean_span_length)

    encoder, decoder = [], [bos]
    sentinel = sentinel_start
    i = 0
    while i < L:
        if mask[i]:
            encoder.append(sentinel)
            decoder.append(sentinel)
            while i < L and mask[i]:
                decoder.append(ids[i])
                i += 1
            sentinel += 1
        else:
            encoder.append(ids[i])
            i += 1
    decoder.append(eos)
    return encoder, decoder

def pack_for_distill(
    examples, tokenizer, pad_id, bos_id, eos_id,
    sentinel_start, seq_len=None, ne=8, nd=8
):
    all_ids = sum(
        tokenizer(examples["text"], add_special_tokens=False)["input_ids"], []
    )

    # make fixed-size chunks
    chunks = [
        all_ids[i : i + seq_len + 1]
        for i in range(0, len(all_ids) - seq_len, seq_len)
    ]

    student_enc, student_dec_in, student_labels = [], [], []
    teacher_in, teacher_mask = [], []

    for chunk in chunks:
        enc_ids, dec_ids = span_corrupt(
            chunk, bos=bos_id, eos=eos_id,
            pad=pad_id, sentinel_start=sentinel_start
        )

        # 🔹 fixed-size encoder (pad up to seq_len+ne)
        s_enc = (enc_ids + [pad_id] * (seq_len + ne - len(enc_ids)))[:seq_len+ne]

        # 🔹 fixed-size decoder inputs & labels
        s_dec_in = (dec_ids[:-1] + [pad_id] * (seq_len + nd - len(dec_ids[:-1])))[:seq_len+nd]
        s_labels = (dec_ids[1:] + [pad_id] * (seq_len + nd - len(dec_ids[1:])))[:seq_len+nd]

        # set loss ignore index for PAD tokens
        s_labels = [tok if tok != pad_id else -100 for tok in s_labels]

        # 🔹 teacher still gets attention mask
        t_in = [pad_id] * ne + enc_ids + dec_ids[1:] + [pad_id] * nd
        t_in = (t_in + [pad_id] * (2*seq_len + ne + nd - len(t_in)))[:2*seq_len+ne+nd]
        t_mask = [0] * len(t_in)
        # mark everything that is not pad
        for j, tok in enumerate(t_in):
            if tok != pad_id:
                t_mask[j] = 1

        student_enc.append(s_enc)
        student_dec_in.append(s_dec_in)
        student_labels.append(s_labels)
        teacher_in.append(t_in)
        teacher_mask.append(t_mask)

    return {
        "student_encoder_input_ids": student_enc,
        "student_decoder_input_ids": student_dec_in,
        "student_labels": student_labels,
        "teacher_input_ids": teacher_in,
        "teacher_attention_mask": teacher_mask,
    }


def load_data(path, tokenizer, seq_len=512, ne=8, nd=8, ratio=0.1):
    dataset = load_dataset("json", data_files=path, split="train")

    total_len = len(dataset)
    val_size = int(total_len * ratio)

    # Validation slice (first 1%)
    val_dataset = dataset.select(range(val_size))

    # Training slice (remaining 99.9%)
    train_dataset = dataset.select(range(val_size, total_len))

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    special_tokens = [f"<extra_id_{i}>" for i in range(100)]
    num_new = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    sentinel_start = tokenizer.convert_tokens_to_ids("<extra_id_0>")

    
    
    train_dataset = train_dataset.map(
        lambda x: pack_for_distill(x, tokenizer, pad, bos, eos, sentinel_start, seq_len, ne, nd),
        batched=True,
        remove_columns=["text"],
    ).with_format("torch")

    val_dataset = val_dataset.map(
        lambda x: pack_for_distill(x, tokenizer, pad, bos, eos, sentinel_start, seq_len, ne, nd),
        batched=True,
        remove_columns=["text"],
    ).with_format("torch")

    return train_dataset, val_dataset, tokenizer
