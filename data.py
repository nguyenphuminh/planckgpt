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

def load_finetune_data(
    start=0,
    end=None,  # None means load full datasets
    step=25000,
    identity_json_path="./data/identity.json"
):
    all_data = []

    # Define datasets with row limits and specific subsets
    datasets_config = [
        ("HuggingFaceTB/smol-smoltalk", "train", None, 460000),
        ("cais/mmlu", "auxiliary_train", None, 100000),
        ("openai/gsm8k", "main", "train", 8000),
    ]

    # Load HuggingFace datasets with row limits
    for config in datasets_config:
        if len(config) == 4:
            dataset_name, subset_name, split, max_rows = config
        else:
            dataset_name, split, max_rows = config
            subset_name = None
        
        print(f"Loading {dataset_name}" + (f" ({subset_name})" if subset_name else "") + f" (max {max_rows} rows)...")
        
        # Load dataset with proper subset handling
        if subset_name:
            dataset = load_dataset(dataset_name, subset_name, split=split) if split else load_dataset(dataset_name, subset_name)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        # Limit to max_rows
        if len(dataset) > max_rows:
            dataset = dataset.select(range(max_rows))
        
        dataset_size = len(dataset)
        print(f"  {dataset_name}: {dataset_size} samples")
        all_data.append(dataset)

    # Load local identity JSON file (3 epochs)
    print(f"Loading {identity_json_path} (3 epochs)...")
    with open(identity_json_path, "r", encoding="utf-8") as f:
        identity_data = json.load(f)
    
    # Repeat identity data 3 times for 3 epochs
    identity_data_3x = identity_data * 3
    
    # Convert to HuggingFace Dataset
    identity_dataset = Dataset.from_list(identity_data_3x)
    identity_size = len(identity_dataset)
    print(f"  {identity_json_path}: {identity_size} samples (3 epochs)")
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
