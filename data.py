import json
from datasets import load_dataset, Dataset, concatenate_datasets

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
    start=4470000,
    end=4480000,
    step=10000,
    dataset_name="HuggingFaceFW/fineweb-edu",
    subset="sample-10BT",
    split="train"
):
    """Data loader for validation"""

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

def load_midtrain_data(
    start=0,
    end=None,  # None means load full datasets
    step=25000,
    identity_json_path="./data/identity.json"
):
    """Data loader for midtraining"""

    all_data = []

    # Define datasets with row limits
    # Format: (dataset_name, config/subset, split, max_rows)
    # config/subset is None if not needed
    datasets_config = [
        ("HuggingFaceTB/smol-smoltalk", None, "train", 460000),
        ("cais/mmlu", "auxiliary_train", "train", 100000),
        ("openai/gsm8k", "main", "train", 8000),
    ]

    # Load HuggingFace datasets with row limits
    for dataset_name, config_name, split, max_rows in datasets_config:
        print(f"Loading {dataset_name}" + (f" ({config_name})" if config_name else "") + f" (max {max_rows} rows)...")
        
        # Load dataset with proper config/subset and split handling
        if config_name and split:
            dataset = load_dataset(dataset_name, config_name, split=split)
        elif config_name:
            dataset = load_dataset(dataset_name, config_name, split="train")
        elif split:
            dataset = load_dataset(dataset_name, split=split)
        else:
            dataset = load_dataset(dataset_name, split="train")
        
        # Limit to max_rows
        if len(dataset) > max_rows:
            dataset = dataset.select(range(max_rows))
        
        dataset_size = len(dataset)
        print(f"  {dataset_name}: {dataset_size} samples")
        all_data.append(dataset)

    # Load local identity JSON file (4 epochs)
    print(f"Loading {identity_json_path} (4 epochs)...")
    with open(identity_json_path, "r", encoding="utf-8") as f:
        identity_data = json.load(f)
    
    # Repeat identity data 4 times for 4 epochs
    identity_data_4x = [{"messages": conv} for conv in identity_data] * 4
    
    # Convert to HuggingFace Dataset
    identity_dataset = Dataset.from_list(identity_data_4x)
    identity_size = len(identity_dataset)
    print(f"  {identity_json_path}: {identity_size} samples (4 epochs)")
    all_data.append(identity_dataset)

    # Concatenate datasets and shuffle
    combined_dataset = concatenate_datasets(all_data).shuffle(seed=42)
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
            # Handle multi-turn conversation format (messages)
            # Used by: smol-smoltalk, identity data
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
            
            # Handle MMLU format (question + choices)
            elif "question" in sample and "choices" in sample:
                question = sample.get("question", "")
                choices = sample.get("choices", [])
                answer = sample.get("answer", 0)
                
                # Format choices
                choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                user_prompt = f"{question}\n{choices_text}"
                
                # Convert answer index to letter
                if isinstance(answer, int):
                    answer_letter = chr(65 + answer)
                else:
                    answer_letter = answer
                
                conversation_text = f"User: {user_prompt}\nAssistant: The answer is {answer_letter}.\n"
                conversation_parts.append(f"{conversation_text}<|endoftext|>\n\n")
            
            # Handle GSM8K format (question + answer with solution)
            elif "question" in sample and "answer" in sample and "choices" not in sample:
                question = sample.get("question", "")
                answer = sample.get("answer", "")
                
                conversation_text = f"User: {question}\nAssistant: {answer}\n"
                conversation_parts.append(f"{conversation_text}<|endoftext|>\n\n")

        combined_text = "".join(conversation_parts)
        
        yield combined_text

def load_sft_data(
    start=0,
    end=None,  # None means load full datasets
    step=25000,
    identity_json_path="./data/identity.json"
):
    all_data = []

    # Define datasets with row limits
    # Format: (dataset_name, config/subset, split, max_rows)
    # config/subset is None if not needed
    datasets_config = [
        ("allenai/ai2_arc", "ARC-Easy", "train", 2300),
        ("allenai/ai2_arc", "ARC-Challenge", "train", 1100),
        ("openai/gsm8k", "main", "train", 8000),
        ("HuggingFaceTB/smol-smoltalk", None, "train", 10000),
    ]

    # Load HuggingFace datasets with row limits
    for dataset_name, config_name, split, max_rows in datasets_config:
        print(f"Loading {dataset_name}" + (f" ({config_name})" if config_name else "") + f" (max {max_rows} rows)...")
        
        # Load dataset with proper config/subset and split handling
        if config_name and split:
            dataset = load_dataset(dataset_name, config_name, split=split)
        elif config_name:
            dataset = load_dataset(dataset_name, config_name, split="train")
        elif split:
            dataset = load_dataset(dataset_name, split=split)
        else:
            dataset = load_dataset(dataset_name, split="train")
        
        # Limit to max_rows
        if len(dataset) > max_rows:
            dataset = dataset.select(range(max_rows))
        
        dataset_size = len(dataset)
        print(f"  {dataset_name}: {dataset_size} samples")
        all_data.append(dataset)

    # Load local identity JSON file (2 epochs)
    print(f"Loading {identity_json_path} (2 epochs)...")
    with open(identity_json_path, "r", encoding="utf-8") as f:
        identity_data = json.load(f)
    
    # Use identity data for 2 epoch (no repetition)
    identity_data_2x = [{"messages": conv} for conv in identity_data]
    
    # Convert to HuggingFace Dataset
    identity_dataset = Dataset.from_list(identity_data_2x)
    identity_size = len(identity_dataset)
    print(f"  {identity_json_path}: {identity_size} samples (2 epochs)")
    all_data.append(identity_dataset)

    # Concatenate datasets and shuffle
    combined_dataset = concatenate_datasets(all_data).shuffle(seed=42)
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
            # Handle multi-turn conversation format (messages)
            # Used by: smol-smoltalk, identity data
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
            
            # Handle ARC format (question + choices + answerKey)
            # Used by: ARC-Easy, ARC-Challenge
            elif "question" in sample and "choices" in sample and "answerKey" in sample:
                question = sample.get("question", "")
                choices_data = sample.get("choices") or {}  # Handle None case
                answer_key = sample.get("answerKey", "")
                
                # ARC format has choices as {"text": [...], "label": [...]}
                choice_texts = choices_data.get("text", [])
                choice_labels = choices_data.get("label", [])
                
                # Format choices with their labels
                choices_text = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)])
                user_prompt = f"{question}\n{choices_text}"
                
                conversation_text = f"User: {user_prompt}\nAssistant: The answer is {answer_key}.\n"
                conversation_parts.append(f"{conversation_text}<|endoftext|>\n\n")
            
            # Handle GSM8K format (question + answer with solution)
            elif "question" in sample and "answer" in sample:
                question = sample.get("question", "")
                answer = sample.get("answer", "")
                
                conversation_text = f"User: {question}\nAssistant: {answer}\n"
                conversation_parts.append(f"{conversation_text}<|endoftext|>\n\n")

        combined_text = "".join(conversation_parts)
        
        yield combined_text
