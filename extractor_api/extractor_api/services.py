import os
import re
import json
import logging
from typing import List, Dict, Tuple
from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from extractor_api.extract_title import extract_titles_from_pdf

# Configure logging for debug purposes.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of unwanted substrings or regex patterns to remove from the markdown.
# You can extend or update this list as needed.
unwanted_substrings: List[str] = [
    # Example: r"some_pattern_to_remove",
]

def init_app():
    """Initialization logic for the application if needed."""
    pass

def extract_pdf_content(
    pdf_file_path: str,
    output_dir: str,
    method: str = 'txt',  # Default to text-based extraction
    lang: str = "it"      # Language for OCR (if needed)
) -> Tuple[str, str, List[Dict]]:
    """
    Extract content from a PDF and store necessary results in the output directory.

    Args:
        pdf_file_path (str): Path to the PDF file.
        output_dir (str): Directory where outputs will be stored.
        method (str, optional): Extraction method to use. Defaults to 'txt'.
        lang (str, optional): Language for OCR. Defaults to "it".

    Returns:
        Tuple[str, str, List[Dict]]: Paths to the markdown file, content list JSON, and the extracted titles.
    """
    # Ensure the output directories exist.
    try:
        image_output_dir = os.path.join(output_dir, "images")
        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating output directories: {e}")
        raise

    # Set up readers and writers.
    try:
        reader = FileBasedDataReader("")
        image_writer = FileBasedDataWriter(image_output_dir)
        md_writer = FileBasedDataWriter(output_dir)
        pdf_bytes = reader.read(pdf_file_path)
    except Exception as e:
        logger.error(f"Error reading PDF file {pdf_file_path}: {e}")
        raise

    # Extract titles using an external extractor.
    try:
        titles = extract_titles_from_pdf(pdf_file_path)
    except Exception as e:
        logger.warning(f"Failed to extract titles from PDF: {e}")
        titles = []  # Fallback to an empty list if extraction fails.

    # Initialize the document dataset.
    try:
        ds = PymuDocDataset(pdf_bytes)
    except Exception as e:
        logger.error(f"Error initializing document dataset: {e}")
        raise

    # Analyze the document based on its classification.
    try:
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True, lang=lang)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False, lang=lang)
            pipe_result = infer_result.pipe_txt_mode(image_writer)
    except Exception as e:
        logger.error(f"Error during document analysis: {e}")
        raise

    # Prepare output file paths.
    pdf_basename = os.path.splitext(os.path.basename(pdf_file_path))[0]
    markdown_file_path = os.path.join(output_dir, f"{pdf_basename}.md")
    content_list_path = os.path.join(output_dir, f"{pdf_basename}_content_list.json")
    qdrant_output_path = os.path.join(output_dir, f"{pdf_basename}_qdrant.json")

    # Save the extracted content.
    try:
        pipe_result.dump_md(md_writer, f"{pdf_basename}.md", image_output_dir)
        pipe_result.dump_content_list(md_writer, f"{pdf_basename}_content_list.json", image_output_dir)
        pipe_result.draw_layout(os.path.join(output_dir, f"{pdf_basename}_layout.pdf"))
    except Exception as e:
        logger.error(f"Error saving extracted data: {e}")
        raise

    # Save data in a Qdrant-friendly format.
    qdrant_data = {
        "markdown_file": markdown_file_path,
        "content_list": content_list_path,
        "titles": titles
    }
    try:
        with open(qdrant_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(qdrant_data, json_file, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Error writing Qdrant output file: {e}")
        raise

    # Read the generated Markdown content.
    try:
        with open(markdown_file_path, 'r', encoding='utf-8') as md_file:
            markdown_content = md_file.read()
    except FileNotFoundError:
        logger.error(f"Markdown file {markdown_file_path} not found.")
        return markdown_file_path, content_list_path, titles
    except Exception as e:
        logger.error(f"Error reading Markdown file {markdown_file_path}: {e}")
        raise

    # Remove unwanted substrings from the markdown.
    if unwanted_substrings:
        for pattern in unwanted_substrings:
            try:
                markdown_content = re.sub(pattern, '', markdown_content, flags=re.DOTALL | re.MULTILINE)
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")

    # Clean up extra blank lines.
    markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)

    try:
        with open(markdown_file_path, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)
    except Exception as e:
        logger.error(f"Error writing cleaned Markdown file {markdown_file_path}: {e}")
        raise

    logger.info(f"Extraction completed. Results are stored in {output_dir}")
    return markdown_file_path, content_list_path, titles

def parse_markdown_with_pages(
    markdown_content: str,
    content_list_path: str,
    titles: List[Dict]
) -> List[Dict]:
    """
    Parse markdown content into sections annotated with titles and page numbers.

    Args:
        markdown_content (str): The complete markdown content.
        content_list_path (str): Path to the JSON file containing the content list.
        titles (List[Dict]): List of extracted title entries.

    Returns:
        List[Dict]: A list of sections with keys "title", "content", and "page_idx".
    """
    sections = []
    current_section = {"title": None, "content": "", "page_idx": None}
    content_list = []

    try:
        if os.path.exists(content_list_path):
            with open(content_list_path, 'r', encoding='utf-8') as file:
                content_list = json.load(file)
    except Exception as e:
        logger.warning(f"Error loading content list from {content_list_path}: {e}")

    # Build mappings for quick lookup of page indices.
    page_map = {item["text"].strip(): item.get("page_idx") 
                for item in content_list if item.get("text_level") == 1}
    extracted_titles = {title_entry["text"].strip(): title_entry.get("page") 
                        for title_entry in titles}

    def is_title_line(line: str) -> bool:
        """
        Determine whether a given line qualifies as a title.

        Args:
            line (str): A line of text from the markdown.

        Returns:
            bool: True if the line is a title, False otherwise.
        """
        line = line.strip()
        # Flexible numeric title pattern (e.g., "1.2. Title" or "1 Title").
        title_pattern = r'^\s*\d+(?:\.\d+)*(?:\.)?\s+.*$'

        if line.startswith('#'):
            title_text = line.lstrip('#').strip()
            # Consider as title if it matches the numeric pattern or exists in our mappings.
            if re.match(title_pattern, title_text):
                return True
            if title_text in extracted_titles or title_text in page_map:
                return True
            return False

        if re.match(title_pattern, line):
            return True

        return line in extracted_titles or line in page_map

    def clean_line(line: str) -> str:
        """Clean extra whitespace from a line."""
        return re.sub(r'\s+', ' ', line).strip()

    # Process the markdown content line by line.
    for raw_line in markdown_content.split("\n"):
        cleaned_line = clean_line(raw_line)
        if not cleaned_line:
            continue

        if is_title_line(cleaned_line):
            # If a section is already in progress, save it.
            if current_section["title"] is not None:
                sections.append(current_section)
            title_text = cleaned_line.lstrip('#').strip()
            page_idx = extracted_titles.get(title_text) or page_map.get(title_text)
            current_section = {"title": title_text, "content": "", "page_idx": page_idx}
        else:
            current_section["content"] += cleaned_line + "\n"

    # Append the last section if it has a title.
    if current_section["title"] is not None:
        sections.append(current_section)

    return sections
