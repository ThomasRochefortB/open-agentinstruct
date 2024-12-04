import json
import argparse

def convert_to_chat_format(input_file, output_file):
    # Read the input JSONL file and convert to chat format
    chat_formatted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
                
            try:
                # Parse the JSONL line
                sample = json.loads(line.strip())
                
                # Create the chat format
                chat_messages = [
                    {
                        "role": "user",
                        "content": sample["instruction"]
                    },
                    {
                        "role": "assistant",
                        "content": sample["answer"]
                    }
                ]
                
                chat_formatted_data.append(chat_messages)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line}")
                print(f"Error message: {e}")
                continue
    
    # Write the output JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for messages in chat_formatted_data:
            json_str = json.dumps(messages, ensure_ascii=False)
            f.write(json_str + '\n')

def main():
    parser = argparse.ArgumentParser(description='Convert JSONL to chat format')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file path')
    
    args = parser.parse_args()
    
    print(f"Converting {args.input} to chat format...")
    convert_to_chat_format(args.input, args.output)
    print(f"Conversion complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()