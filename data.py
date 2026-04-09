from datasets import load_dataset

def load_data(
    start=0,
    end=2900360,
    step=145018,
    dataset_name="HuggingFaceFW/fineweb-edu",
    subset="sample-10BT",
    split="train"
):
    """Data loader for pretraining"""

    # Download full dataset (no streaming)
    dataset = load_dataset(dataset_name, name=subset, split=split)
    dataset_size = len(dataset)
    
    print(f"Dataset loaded: {dataset_size} documents")

    # Load segments of the dataset
    for segment_index in range(start, min(end, dataset_size), step):
        if segment_index >= dataset_size:
            return

        segment_end = min(segment_index + step, dataset_size, end)
        miniset = dataset.select(range(segment_index, segment_end))
        text_parts = []

        for sample in miniset:
            text = sample.get("text", "")
            if text:
                text_parts.append(f"{text}<|endoftext|>\n\n")

        combined_text = "".join(text_parts)
        
        yield combined_text

def load_val_data(
    dataset_name="wikitext",
    subset="wikitext-103-raw-v1",
    split="validation"
):
    """Data loader for validation using wikitext"""
    dataset = load_dataset(dataset_name, name=subset, split=split)
    
    text_parts = []
    for sample in dataset:
        text = sample.get("text", "")
        if text.strip():
            text_parts.append(text)
    
    combined_text = "\n\n".join(text_parts)
    yield combined_text
