import csv
import json

def csv_to_jsonl(csv_file, jsonl_file):
    """Convert fraud detection CSV to JSONL format for fine-tuning."""
    
    with open(csv_file, 'r', encoding='utf-8') as f_csv, \
         open(jsonl_file, 'w', encoding='utf-8') as f_jsonl:
        
        reader = csv.DictReader(f_csv)
        
        for row in reader:
            # Extract relevant features for the prompt
            merchant = row.get('merchant', '').replace('fraud_', '')
            category = row.get('category', '')
            amount = row.get('amt', '')
            city = row.get('city', '')
            state = row.get('state', '')
            job = row.get('job', '')
            is_fraud = row.get('is_fraud', '0')
            
            # Create the prompt with transaction details
            prompt = f"""Analyze this transaction for potential fraud:
- Merchant: {merchant}
- Category: {category}
- Amount: ${amount}
- Location: {city}, {state}
- Cardholder occupation: {job}

Is this transaction fraudulent?"""
            
            # Create the completion/response
            if is_fraud == '1':
                completion = "Yes, this transaction appears to be fraudulent. The transaction pattern shows characteristics commonly associated with fraud."
            else:
                completion = "No, this transaction appears to be legitimate. The transaction pattern is consistent with normal purchasing behavior."
            
            # Create JSONL entry in chat format
            entry = {
                "messages": [
                    {"role": "system", "content": "You are a fraud detection assistant. Analyze transactions and determine if they are fraudulent based on the provided details."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ]
            }
            
            f_jsonl.write(json.dumps(entry) + '\n')
    
    print(f"Converted {csv_file} to {jsonl_file}")

if __name__ == "__main__":
    csv_to_jsonl('fraudTest.csv', 'training.jsonl')
    print("Done!")
