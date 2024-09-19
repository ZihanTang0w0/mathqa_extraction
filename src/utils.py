import json


def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8', errors="ignore") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def write_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8', errors="ignore") as file:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False, indent=4)
            file.write(json_line + '\n')

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        buffer = ""
        for line in file:
            buffer += line.strip()
            try:
                # Attempt to parse the JSON
                record = json.loads(buffer)
                data.append(record)
                buffer = ""  # Reset buffer if successful
            except json.JSONDecodeError:
                # If error, continue appending lines
                continue
    return data

def convert_to_markdown(text):
    """
    Convert a given text string to a Markdown formatted string.

    Parameters:
    - text (str): The plain text string to convert.

    Returns:
    - str: The Markdown formatted string.
    """
    # Define replacements in the order they should occur
    replacements = {
        '\n\n': '\n\n',
        '### ': '\n### ',
        '1. ': '\n1. ',
        '2. ': '\n2. ',
        '3. ': '\n3. ',
        '4. ': '\n4. ',
        '5. ': '\n5. ',
        '6. ': '\n6. ',
        '- ': '   - ',
        '**': '**'
    }

    # Perform replacements
    markdown_text = text
    for old, new in replacements.items():
        markdown_text = markdown_text.replace(old, new)

    # Ensuring correct newlines for Markdown sections
    markdown_text = markdown_text.strip()  # Remove leading/trailing whitespace
    return markdown_text

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list