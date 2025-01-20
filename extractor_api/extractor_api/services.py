import os
import re
import json
from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

def init_app():
    """Initialization logic for the application if needed."""
    pass

def extract_pdf_content(pdf_file_path, output_dir):
    """Extract content from a PDF and save results to the output directory."""
    
    # Ensure output directory exists
    image_output_dir = os.path.join(output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    # Set up readers and writers
    reader = FileBasedDataReader("")
    image_writer = FileBasedDataWriter(image_output_dir)
    md_writer = FileBasedDataWriter(output_dir)

    # Read PDF content
    pdf_bytes = reader.read(pdf_file_path)
    ds = PymuDocDataset(pdf_bytes)

    # Determine parsing method and apply the appropriate pipeline
    if ds.classify() == SupportedPdfParseMethod.OCR:
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    # Save outputs
    pdf_basename = os.path.basename(pdf_file_path).split('.')[0]
    markdown_file_path = os.path.join(output_dir, f"{pdf_basename}.md")
    content_list_path = os.path.join(output_dir, f"{pdf_basename}_content_list.json")
    
    pipe_result.dump_md(md_writer, f"{pdf_basename}.md", image_output_dir)
    pipe_result.dump_content_list(md_writer, f"{pdf_basename}_content_list.json", image_output_dir)
    pipe_result.dump_middle_json(md_writer, f'{pdf_basename}_middle.json')
    infer_result.draw_model(os.path.join(output_dir, f"{pdf_basename}_model.pdf"))
    pipe_result.draw_layout(os.path.join(output_dir, f"{pdf_basename}_layout.pdf"))
    pipe_result.draw_span(os.path.join(output_dir, f"{pdf_basename}_spans.pdf"))

    print(f"Extraction completed. Results are stored in {output_dir}")

    return markdown_file_path, content_list_path  # Return both markdown and content list paths

def parse_markdown_with_pages(markdown_content, content_list_path):
    """Parse markdown content into structured JSON with page numbers."""
    
    sections = []
    current_section = {"title": None, "content": "", "page_idx": None}
    lines = markdown_content.split("\n")

    # Load page numbers from the content list JSON file
    if os.path.exists(content_list_path):
        with open(content_list_path, 'r', encoding='utf-8') as json_file:
            content_list = json.load(json_file)
    else:
        content_list = []

    # Map content to corresponding page numbers by trimming whitespace and matching text
    page_map = {item["text"].strip(): item["page_idx"] for item in content_list if item["type"] == "text"}

    for line in lines:
        header_match = re.match(r"^(#{1,6})\s*(.*)", line)
        if header_match:
            # If we have a previous section, store it before starting a new one
            if current_section["title"]:  
                sections.append(current_section)

            # Start a new section
            title = header_match.group(2).strip()
            current_section = {"title": title, "content": "", "page_idx": page_map.get(title, None)}
        else:
            current_section["content"] += line + "\n"

    if current_section["title"]:
        sections.append(current_section)
    
    return sections