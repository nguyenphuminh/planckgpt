import random
import json
from datasets import load_dataset

USER_PREFIX = "User: "
ASSISTANT_PREFIX = "\n\nAssistant:"
# The model is trained to stop with the end of text token, but if it forgets it
# will fall back on starting a new user turn. Inference can cut it off here.
STOP_STRING = "\n\nUser:"
# Token id, not the string: everything else in a row is an id, and rows go
# straight into torch.tensor(..., dtype=torch.long). Must match gpt.eos_token_id.
EOS = 50256

class SmoltalkLoader:
    """Smol-smoltalk data loader utility"""

    def __init__(self, split="train"):
        self.dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {"messages": self.dataset[index]["messages"]}

class IdentityLoader:
    """Identity data loader utility"""

    def __init__(self, epochs=2):
        with open("./data/finetune_identity.jsonl", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()] * epochs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

def normalize(messages):
    """
    Fold an optional leading system message into the first user message, then
    check the conversation is user/assistant alternating and ends on assistant.
    Returns None for anything malformed so the caller can skip it.
    """

    messages = [dict(message) for message in messages]

    if messages and messages[0]["role"] == "system":
        if len(messages) < 2 or messages[1]["role"] != "user":
            return None

        messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
        messages = messages[1:]

    if len(messages) < 2 or len(messages) % 2 != 0:
        return None

    for i, message in enumerate(messages):
        if message["role"] != ("user" if i % 2 == 0 else "assistant"):
            return None
        if not isinstance(message["content"], str):
            return None

    return messages

def render(encode, messages, max_tokens):
    """
    Render one conversation into (ids, mask), where mask[i] == 1 means token i
    is a target the assistant is trained to produce:

        User: {user}\n\nAssistant: {assistant}<eos>User: {user}\n\n...

    Turns are added whole. If the next turn would overflow max_tokens the
    conversation is cut short there, so the model never sees an assistant reply
    whose question was truncated away. Returns (None, None) if not even the
    first turn fits.
    """

    messages = normalize(messages)

    if messages is None:
        return None, None

    ids = []
    mask = []

    for i in range(0, len(messages), 2):
        prompt_ids = encode(USER_PREFIX + messages[i]["content"] + ASSISTANT_PREFIX)
        # Leading space so the first word merges the way it does in normal text:
        # "Assistant:" + " Hello" is how the tokenizer splits "Assistant: Hello"
        answer_ids = encode(" " + messages[i + 1]["content"])

        if len(ids) + len(prompt_ids) + len(answer_ids) + 1 > max_tokens:
            break

        ids += prompt_ids
        mask += [0] * len(prompt_ids)   # The user's turn is context, not a target
        ids += answer_ids
        mask += [1] * len(answer_ids)
        ids.append(EOS)
        mask.append(1)                  # Supervised: this is how it learns to stop

    if not ids:
        return None, None

    return ids, mask

def pack(conversations, batch_size, row_capacity, total, buffer_size=128):
    """
    Best-fit packing. Rows are filled with whole conversations, never split, and
    whatever space is left over at the end is padded rather than crammed.

    `conversations` is an iterator of (ids, mask). Yields (rows, masks, progress)
    where rows and masks are lists of batch_size lists of exactly row_capacity
    ints, and progress runs 0 -> 1 over the pass. The training loop cannot know
    the step count in advance, since it depends on how the conversations happen
    to pack, so it drives its LR schedule off progress instead.
    """

    buffer = []
    exhausted = False
    consumed = 0

    while True:
        rows = []
        masks = []

        for _ in range(batch_size):
            row = []
            row_mask = []

            while len(row) < row_capacity:
                # Keep the buffer topped up so best-fit has something to choose from
                while not exhausted and len(buffer) < buffer_size:
                    item = next(conversations, None)

                    if item is None:
                        exhausted = True
                        break

                    buffer.append(item)

                if not buffer:
                    break

                # Pick the largest conversation that still fits in the gap
                remaining = row_capacity - len(row)
                best = -1
                best_len = 0

                for index, (candidate_ids, _) in enumerate(buffer):
                    if best_len < len(candidate_ids) <= remaining:
                        best = index
                        best_len = len(candidate_ids)

                if best < 0:
                    break   # Nothing fits, pad the rest below

                ids, mask = buffer.pop(best)
                row += ids
                row_mask += mask
                consumed += 1

            if not row:
                return  # Everything consumed

            pad = row_capacity - len(row)

            if pad > 0:
                row += [EOS] * pad
                row_mask += [0] * pad   # Masked, or the model just learns to emit eos

            rows.append(row)
            masks.append(row_mask)

        if len(rows) < batch_size:
            return  # Drop the last incomplete batch, keeps shapes static for compile

        yield rows, masks, min(consumed / total, 1.0)

def build(datasets, encode, batch_size, seq_len, seed=42):
    """Shuffle every conversation across all datasets together, then pack."""

    index_map = []

    for i in range(0, len(datasets)):
        index_map += [(i, j) for j in range(0, len(datasets[i]))]

    random.Random(seed).shuffle(index_map)

    row_capacity = seq_len + 1   # +1 because targets are the inputs shifted by one

    def conversations():
        for i, j in index_map:
            ids, mask = render(encode, datasets[i][j]["messages"], row_capacity)

            if ids is not None:
                yield ids, mask

    return pack(conversations(), batch_size, row_capacity, max(len(index_map), 1))

def load_data(encode, batch_size=8, seq_len=1024):
    """Data loader for chat finetuning"""

    datasets = [
        SmoltalkLoader(split="train"),
        IdentityLoader()
    ]

    return build(datasets, encode, batch_size, seq_len)

def load_val_data(encode, batch_size=8, seq_len=1024):
    """Data loader for chat finetuning validation"""

    datasets = [
        SmoltalkLoader(split="test")
    ]

    return build(datasets, encode, batch_size, seq_len)
