import torch
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to SLM")
    parser.add_argument("--layer_idx", type=int, default=-1, help="Which layer to probe (-1 for the last layer)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset_path", type=str, required=True, help="HF dataset name (e.g., 'gsm8k') or path to local file (e.g., /scratch/.../data.json)")
    parser.add_argument("--text_col", type=str, default="text", help="The column name containing the input text")
    parser.add_argument("--label_col", type=str, default="label", help="The column name containing the classification label")
    return parser.parse_args()


def extract_hidden_states(model, tokenizer, texts, layer_idx, device):
    """
    Passes text through the frozen model and extracts the hidden state 
    of the final token at the specified layer.
    """
    model.eval()
    hidden_states = []
    
    print(f"Extracting representations from layer {layer_idx} on {device} for the last token...")
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            
            # Extract the specified layer
            layer_hidden_states = outputs.hidden_states[layer_idx]
            
            # Grab the representation of the last token (change if you want, but last token is probably most informative) 
            last_token_state = layer_hidden_states[0, -1, :].cpu().numpy()
            hidden_states.append(last_token_state)
            
    return np.array(hidden_states)

def main():
    args = get_args()
    
    print(f"Loading dataset from: {args.dataset_path}...") 
    if args.dataset_path.endswith(('.csv', '.json', '.jsonl')):
        dataset = load_dataset(args.dataset_path.split('.')[-1], data_files=args.dataset_path)
        dataset = dataset['train'] # Default to train split for local files
    else:
        dataset = load_dataset(args.dataset_path, split="train") 
    texts = dataset[args.text_col]
    labels = np.array(dataset[args.label_col])
    
    print(f"Loaded {len(texts)} samples. Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(args.device)
    X = extract_hidden_states(model, tokenizer, texts, args.layer_idx, args.device)
    y = labels

    print("Training linear probe...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    probe = LogisticRegression(max_iter=1000)
    probe.fit(X_train, y_train)
    
    predictions = probe.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print(f"\n--- Probing Results (Layer {args.layer_idx}) ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, predictions, zero_division=0))

if __name__ == "__main__":
    main()
