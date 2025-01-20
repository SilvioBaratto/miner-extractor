import json
import re

# Load the JSON content
with open('miner_extractor/example.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

content = data["content"]

# Function to parse the markdown content
def parse_markdown(content):
    sections = []
    current_section = {"title": None, "content": ""}
    lines = content.split("\n")
    
    for line in lines:
        # Detect headers based on markdown syntax
        header_match = re.match(r"^(#{1,6})\s*(.*)", line)
        if header_match:
            if current_section["title"]:  
                sections.append(current_section)
            current_section = {"title": header_match.group(2).strip(), "content": ""}
        else:
            current_section["content"] += line + "\n"
    
    if current_section["title"]:
        sections.append(current_section)
    
    return sections

# Parse the markdown content
parsed_sections = parse_markdown(content)

print(parsed_sections)
