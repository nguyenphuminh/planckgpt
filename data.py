import json
from datasets import Dataset, load_dataset, concatenate_datasets

def load_data(
    start=0,
    end=2893480,
    step=144674,
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

def load_finetune_data(
    start=0,
    end=None,  # None means load full datasets
    step=25000,
    datasets=[
        ("HuggingFaceTB/smol-smoltalk", "train"),
        ("yahma/alpaca-cleaned", "train")
    ],
    identity_json_path="./data/identity.json"
):
    all_data = []

    # Load HuggingFace datasets
    for dataset_name, split in datasets:
        print(f"Loading {dataset_name}...")
        dataset = load_dataset(dataset_name, split=split)
        dataset_size = len(dataset)
        print(f"  {dataset_name}: {dataset_size} samples")
        all_data.append(dataset)

    # Load local identity JSON file
    print(f"Loading {identity_json_path}...")
    with open(identity_json_path, "r", encoding="utf-8") as f:
        identity_data = json.load(f)
    
    # Convert to HuggingFace Dataset
    identity_dataset = Dataset.from_list(identity_data)
    identity_size = len(identity_dataset)
    print(f"  {identity_json_path}: {identity_size} samples")
    all_data.append(identity_dataset)

    # Concatenate datasets
    combined_dataset = concatenate_datasets(all_data)
    total_size = len(combined_dataset)
    print(f"\nCombined dataset: {total_size} samples")

    # Set end to full dataset if not specified
    if end is None:
        end = total_size

    # Load segments of the combined dataset
    for segment_index in range(start, min(end, total_size), step):
        if segment_index >= total_size:
            return

        segment_end = min(segment_index + step, total_size, end)
        miniset = combined_dataset.select(range(segment_index, segment_end))
        conversation_parts = []

        for sample in miniset:
            # Handle smol-smoltalk format (messages)
            messages = sample.get("messages", [])
            
            if messages:
                conversation_text = ""
                for message in messages:
                    role = message.get("role", "")
                    content = message.get("content", "")
                    
                    if role == "user":
                        conversation_text += f"User: {content}\n"
                    elif role == "assistant":
                        conversation_text += f"Assistant: {content}\n"
                
                if conversation_text:
                    conversation_parts.append(f"{conversation_text}<|endoftext|>\n\n")
            
            # Handle alpaca-cleaned and identity format (instruction/input/output)
            elif "instruction" in sample:
                instruction = sample.get("instruction", "")
                input_text = sample.get("input", "")
                output = sample.get("output", "")
                
                user_prompt = instruction
                if input_text:
                    user_prompt += f"\n{input_text}"
                
                conversation_text = f"User: {user_prompt}\nAssistant: {output}\n"
                conversation_parts.append(f"{conversation_text}<|endoftext|>\n\n")

        combined_text = "".join(conversation_parts)
        
        yield combined_text
