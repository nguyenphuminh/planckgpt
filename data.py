from datasets import load_dataset

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
    end=484570,
    step=24229,
    dataset_name="HuggingFaceTB/smol-smoltalk",
    split="train"
):
    # Download full dataset (no streaming)
    dataset = load_dataset(dataset_name, split=split)
    dataset_size = len(dataset)
    
    print(f"Dataset loaded: {dataset_size} samples")

    # Load segments of the dataset
    for segment_index in range(start, min(end, dataset_size), step):
        if segment_index >= dataset_size:
            return

        segment_end = min(segment_index + step, dataset_size, end)
        miniset = dataset.select(range(segment_index, segment_end))
        conversation_parts = []

        for sample in miniset:
            messages = sample.get("messages", [])

            if not messages:
                continue
                
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

        combined_text = "".join(conversation_parts)
        
        yield combined_text
