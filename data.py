import json
from datasets import load_dataset, Dataset, concatenate_datasets

def load_data(
    start=0,
    end=4470000,
    step=223500,
    dataset_name="HuggingFaceFW/fineweb-edu",
    subset="sample-10BT",
    split="train"
):
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
    start=4470000,
    end=4485000,
    step=15000,
    dataset_name="HuggingFaceFW/fineweb-edu",
    subset="sample-10BT",
    split="train"
):
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

def _format_conversation_sample(sample):
    """Format a single sample into conversation text."""
    
    # Handle multi-turn conversation format (messages)
    # Used by: smol-smoltalk, identity data
    if "messages" in sample and sample["messages"]:
        conversation_text = ""
        for message in sample["messages"]:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "user":
                conversation_text += f"User: {content}\n"
            elif role == "assistant":
                conversation_text += f"Assistant: {content}\n"
        
        return f"{conversation_text}<|endoftext|>\n\n" if conversation_text else ""
    
    # Handle MMLU format (question + choices + answer as index)
    if "question" in sample and "choices" in sample and "answer" in sample:
        question = sample["question"]
        choices = sample["choices"]
        answer = sample["answer"]
        
        # Format choices with letters
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        user_prompt = f"{question}\n{choices_text}"
        
        # Convert answer to letter if it's an index
        answer_letter = chr(65 + answer) if isinstance(answer, int) else answer
        
        conversation = f"User: {user_prompt}\nAssistant: The answer is {answer_letter}.\n"
        return f"{conversation}<|endoftext|>\n\n"
    
    # Handle ARC format (question + choices dict + answerKey)
    if "question" in sample and "choices" in sample and "answerKey" in sample:
        question = sample["question"]
        choices_data = sample.get("choices") or {}
        answer_key = sample["answerKey"]
        
        # ARC format: {"text": [...], "label": [...]}
        choice_texts = choices_data.get("text", [])
        choice_labels = choices_data.get("label", [])
        
        choices_text = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)])
        user_prompt = f"{question}\n{choices_text}"
        
        conversation = f"User: {user_prompt}\nAssistant: The answer is {answer_key}.\n"
        return f"{conversation}<|endoftext|>\n\n"
    
    # Handle GSM8K format (question + answer with solution, no choices)
    if "question" in sample and "answer" in sample:
        question = sample["question"]
        answer = sample["answer"]
        
        conversation = f"User: {question}\nAssistant: {answer}\n"
        return f"{conversation}<|endoftext|>\n\n"
    
    return ""

def _load_hf_dataset(dataset_name, config_name, split, max_rows):
    """Load a HuggingFace dataset with proper handling of config/split."""
    print(f"Loading {dataset_name}" + (f" ({config_name})" if config_name else "") + f" (max {max_rows} rows)...")
    
    # Load with appropriate parameters
    if config_name and split:
        dataset = load_dataset(dataset_name, config_name, split=split)
    elif config_name:
        dataset = load_dataset(dataset_name, config_name, split="train")
    elif split:
        dataset = load_dataset(dataset_name, split=split)
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    # Limit rows
    if len(dataset) > max_rows:
        dataset = dataset.select(range(max_rows))
    
    print(f"  {dataset_name}: {len(dataset)} samples")
    return dataset

def _load_identity_data(identity_json_path, epochs):
    """Load and repeat identity data for specified epochs."""
    print(f"Loading {identity_json_path} ({epochs} epochs)...")
    
    with open(identity_json_path, "r", encoding="utf-8") as f:
        identity_data = json.load(f)
    
    # Repeat for specified epochs
    identity_data_repeated = [{"messages": conv} for conv in identity_data] * epochs
    
    identity_dataset = Dataset.from_list(identity_data_repeated)
    print(f"  {identity_json_path}: {len(identity_dataset)} samples ({epochs} epochs)")
    
    return identity_dataset

def _load_and_process_data(datasets_config, identity_json_path, identity_epochs, start, end, step):
    """Common data loading and processing logic."""
    all_data = []
    
    # Load HuggingFace datasets
    for dataset_name, config_name, split, max_rows in datasets_config:
        dataset = _load_hf_dataset(dataset_name, config_name, split, max_rows)
        all_data.append(dataset)
    
    # Load identity data
    identity_dataset = _load_identity_data(identity_json_path, identity_epochs)
    all_data.append(identity_dataset)
    
    # Combine and shuffle
    combined_dataset = concatenate_datasets(all_data).shuffle(seed=42)
    total_size = len(combined_dataset)
    print(f"\nCombined dataset: {total_size} samples")
    
    # Set end to full dataset if not specified
    if end is None:
        end = total_size
    
    # Yield segments
    for segment_index in range(start, min(end, total_size), step):
        if segment_index >= total_size:
            return
        
        segment_end = min(segment_index + step, total_size, end)
        miniset = combined_dataset.select(range(segment_index, segment_end))
        
        # Format all samples
        conversation_parts = [
            _format_conversation_sample(sample)
            for sample in miniset
        ]
        
        combined_text = "".join(filter(None, conversation_parts))
        yield combined_text

def load_midtrain_data(
    start=0,
    end=None,
    step=25000,
    identity_json_path="./data/identity.json"
):
    """Load mid-training data with 4 epochs of identity data."""
    datasets_config = [
        ("HuggingFaceTB/smol-smoltalk", None, "train", 460000),
        ("cais/mmlu", "auxiliary_train", "train", 100000),
        ("openai/gsm8k", "main", "train", 8000),
    ]
    
    yield from _load_and_process_data(
        datasets_config, 
        identity_json_path, 
        identity_epochs=4,
        start=start,
        end=end,
        step=step
    )

def load_sft_data(
    start=0,
    end=None,
    step=25000,
    identity_json_path="./data/identity.json"
):
    """Load SFT data with 2 epochs of identity data."""
    datasets_config = [
        ("allenai/ai2_arc", "ARC-Easy", "train", 2300),
        ("allenai/ai2_arc", "ARC-Challenge", "train", 1100),
        ("openai/gsm8k", "main", "train", 8000),
        ("HuggingFaceTB/smol-smoltalk", None, "train", 10000),
    ]
    
    yield from _load_and_process_data(
        datasets_config,
        identity_json_path,
        identity_epochs=2,
        start=start,
        end=end,
        step=step
    )
