import os
import glob
import re
import statistics
import string

import fitz  # PyMuPDF
import pdfplumber

def _line_in_table(
        line_bbox, 
        table_bboxes
    ) -> bool:
    """
    Check if a line intersects with any table bounding box.
    """
    x0, y0, x1, y1 = line_bbox
    for (tx0, ttop, tx1, tbottom) in table_bboxes:
        # Boxes overlap if they do not lie completely outside each other
        if not (x1 < tx0 or x0 > tx1 or y1 < ttop or y0 > tbottom):
            return True
    return False

def _mark_lines_in_tables(
        lines, 
        table_bboxes
    ) -> None:
    """
    Marks lines that intersect with any table bounding box as 'in_table'.
    """

    # Segna ogni riga che si trova all'interno di una tabella.
    # Controlla se il bounding box di una riga si sovrappone con il bounding box di una tabella.
    for line in lines:
        if _line_in_table(line['bbox'], table_bboxes):
            line['in_table'] = True

def _extract_page_lines(
        page
    ) -> list:
    """
    Extract text lines and associated metadata from a PDF page.
    """
    text_data = page.get_text("dict")
    lines = []
    for block in text_data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            span_texts = [span["text"].strip() for span in spans if span["text"].strip()]
            line_text = " ".join(span_texts)

            if not line_text:
                continue

            font_sizes = [span["size"] for span in spans]
            avg_font_size = statistics.mean(font_sizes) if font_sizes else 0
            font_names = [span["font"] for span in spans]
            bbox = line["bbox"]

            lines.append({
                'text': line_text,
                'size': avg_font_size,
                'fontnames': font_names,
                'bbox': bbox
            })
    return lines

def _ends_with_punctuation(
        text
    ) -> bool:
    """
    Check if a text ends with punctuation.
    """
    return bool(re.match(r'.*[.?!:;]$', text.strip()))

def _is_likely_title_format(
        line_dict
    ) -> bool:
    """
    Determine if a line is likely to be a title based on its formatting (bold or uppercase ratio).
    """
    text = line_dict['text']
    bold = any(('Bold' in fn or 'Black' in fn or 'Heavy' in fn) for fn in line_dict['fontnames'])
    letters = [c for c in text if c.isalpha()]
    uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0
    return bold or uppercase_ratio > 0.3

def _find_main_title(lines, page_height, page_width) -> dict:
    """
    Attempt to find the main title from the first page by looking for the largest text lines
    near the top of the page and merging consecutive lines that likely belong together.
    """
    lines_sorted = sorted(lines, key=lambda x: x['size'], reverse=True)
    top_cutoff = page_height * 0.6  # Consider only lines in the top 60% of the page

    # Filter lines that are reasonably sized and positioned near the top of the page
    candidate_lines = []
    for ln in lines_sorted:
        txt = ln['text'].strip()
        if len(txt) > 5 and not _ends_with_punctuation(txt):  # Avoid too-short or punctuated lines
            top_y = ln['bbox'][1]
            if top_y < top_cutoff and _is_likely_title_format(ln):
                candidate_lines.append(ln)

    # Refine merging logic
    merged_lines = []
    if candidate_lines:
        merged_text = candidate_lines[0]['text']
        merged_bbox = list(candidate_lines[0]['bbox'])  # Convert to list for merging
        for i in range(1, len(candidate_lines)):
            current_line = candidate_lines[i]
            previous_line = candidate_lines[i - 1]

            # Calculate vertical and horizontal gaps
            vertical_gap = current_line['bbox'][1] - previous_line['bbox'][3]

            # Ensure gaps are within reasonable limits to avoid unrelated lines
            if (
                0 < vertical_gap < 20  # Vertical gap tolerance: lines must be close
                and current_line['size'] == previous_line['size']  # Ensure font sizes match
                and set(current_line['fontnames']) == set(previous_line['fontnames'])  # Font consistency
            ):
                # Merge text and expand bounding box
                merged_text += f" {current_line['text']}"
                merged_bbox[2] = max(merged_bbox[2], current_line['bbox'][2])  # Update right edge
                merged_bbox[3] = current_line['bbox'][3]  # Update bottom edge
            else:
                # Save the merged line and start a new group
                merged_lines.append({
                    'text': merged_text.strip(),
                    'bbox': merged_bbox,
                    'size': previous_line['size'],
                    'fontnames': previous_line['fontnames'],
                })
                merged_text = current_line['text']
                merged_bbox = list(current_line['bbox'])

        # Add the last merged line
        merged_lines.append({
            'text': merged_text.strip(),
            'bbox': merged_bbox,
            'size': candidate_lines[-1]['size'],
            'fontnames': candidate_lines[-1]['fontnames'],
        })

    # Return the merged line with the largest font size as the main title
    if merged_lines:
        return max(merged_lines, key=lambda x: x['size'])
    elif candidate_lines:
        return candidate_lines[0]
    return None

def _is_bullet_point(text: str) -> bool:
    """
    Detect if the text is a standalone bullet.
    """
    bullet_characters = ['-', '•', '▪', '●']
    return text.strip() in bullet_characters

def _is_potential_bullet_line(text: str) -> bool:
    """
    Detect if a line looks like a bullet line.
    Examples:
    - PHPStone download
    - • Google Chrome
    """
    bullet_pattern = r'^[\u2022\u25AA\u2023\u25B8\u2219\-]\s*.+$'
    return bool(re.match(bullet_pattern, text.strip()))

def _is_section_number(text) -> bool:
    """
    Check if text represents a section number (e.g., "1.2.3", "2.", "2").
    """
    pattern = r'^\d+(\.\d+)*\.?$'
    return bool(re.match(pattern, text.strip()))

def _clean_text(text) -> str:
    """
    Clean extracted text by removing non-printable characters and excessive whitespace.
    """
    text = ''.join([c if c.isprintable() else ' ' for c in text]).strip()
    text = ' '.join(text.split())
    return text.strip()

import re

def _is_part_of_paragraph(current_line, prev_line, next_line):
    """
    Determines if a line is part of a paragraph.
    It is considered part of a paragraph if:
      - It is close in vertical position to the line before or after.
      - It contains too many words.
      - It is not a section number like 1.1.1 or 2.3.4.
      - It is not a bold, short title.
    """
    # 1. Check if it's a section number (like 1.1.1 Ambito di applicabilità)
    section_pattern = r'^\d+(\.\d+)*\s*.+$'
    if re.match(section_pattern, current_line['text']):
        return False  # Section numbers are NOT paragraphs

    # 2. Calculate the distance from the previous and next lines
    prev_distance = current_line['bbox'][1] - prev_line['bbox'][3] if prev_line else float('inf')
    next_distance = next_line['bbox'][1] - current_line['bbox'][3] if next_line else float('inf')

    # 3. If the distance between lines is small, it is part of a paragraph
    close_to_others = prev_distance < 20 or next_distance < 20

    # 4. If the line has more than 10 words, it is likely part of a paragraph
    is_long_line = len(current_line['text'].split()) > 10

    return close_to_others

def _is_potential_title(text, font_size, fontnames, median_font_size, size_factor=1.0, bbox=None, prev_line=None, next_line=None, page_width=None) -> bool:
    """
    Check if text is a potential title based on its format and size.
    Titles should:
    - Be bold and larger than a specified threshold
    - Exclude specific non-title phrases, bullet point items, and patterns
    - Check if it's near the top of the page or in a prominent position
    - Exclude lines that look like part of lists or bullet points
    """
    # Exclude specific phrases
    excluded_phrases = [
        "Identificativo Documento:", "Ambiti :", 
        "Classificazione di riservatezza :", "Destinatari :",
        "Documento e informazioni per circolazione e uso esclusivamente interni", 
    ]
    
    if text.strip() in excluded_phrases:
        return False

    # Exclude text that matches specific ID-like patterns
    id_pattern = r'^ID:\s*[A-Za-z0-9_-]+$'  # Matches "ID: xyz", "ID: IS-01", etc.
    if re.match(id_pattern, text.strip()):
        return False

    # Exclude lines that follow bullet point symbols (like "-", "•", "▪")
    if _is_bullet_point(text) or _is_potential_bullet_line(text):
        return False

    # Exclude lines with specific patterns indicating bullet-style lists
    bullet_point_pattern = r'^[\u2022\u25AA\u2023\u25B8\u2219\-]\s*.+$'
    if re.match(bullet_point_pattern, text.strip()):
        return False

    # Check if it looks like a section number
    section_pattern = r'^\d+(\.\d+)*\s*.+$'
    is_section_number = bool(re.match(section_pattern, text))

    # Check if the font is bold
    is_bold = any('Bold' in font or 'Black' in font or 'Heavy' in font for font in fontnames)

    # Check if the font size is sufficiently large
    is_large = font_size >= median_font_size * size_factor

    # Check if the line is "wide" enough to be a title (avoids short text like "• R")
    if bbox:
        width = bbox[2] - bbox[0]
        if width < 50:  # Arbitrary threshold
            return False
        
    # Check if the current line is part of a paragraph
    if prev_line or next_line:
        if _is_part_of_paragraph({'bbox': bbox, 'text': text}, prev_line, next_line):
            return False

    # Title is valid if it's a section number or bold and large
    return is_section_number or (is_bold and is_large)

def _find_title_candidates(lines, median_font_size, page_number, size_factor=1.0) -> list:
    """
    Identify candidate title lines based on font size, formatting, and proximity to the top.
    """
    candidates = []
    skip_next_line = False

    for i, ln in enumerate(lines):
        if skip_next_line:
            skip_next_line = False
            continue

        # Extract metadata from the line
        text = ln['text'].strip()
        font_size = ln['size']
        fontnames = ln['fontnames']
        bbox = ln['bbox']

        prev_line = lines[i-1] if i > 0 else None
        next_line = lines[i+1] if i + 1 < len(lines) else None

        # Check if the line is sufficiently large and looks like a title
        is_large = font_size >= median_font_size * size_factor
        looks_like_title = _is_potential_title(text, font_size, fontnames, median_font_size, size_factor, bbox, prev_line, next_line)

        # Check if the previous line is a bullet point and if this line follows it
        if i > 0:
            previous_line_text = lines[i-1]['text'].strip()
            if _is_bullet_point(previous_line_text) or _is_potential_bullet_line(previous_line_text):
                # If the previous line is a bullet, this is not a title
                continue

        # Merge section number with the next line if needed
        if _is_section_number(text) and i + 1 < len(lines):
            next_line = lines[i + 1]
            if not _is_section_number(next_line['text']):
                merged_text = f"{text} {next_line['text']}"
                merged_text = _clean_text(merged_text)
                merged_line = {
                    'text': merged_text,
                    'size': max(font_size, next_line['size']),
                    'fontnames': fontnames + next_line['fontnames'],
                    'bbox': ln['bbox']
                }

                if (merged_line['size'] >= median_font_size * size_factor or
                        _is_potential_title(merged_line['text'], merged_line['size'], merged_line['fontnames'], median_font_size, size_factor, merged_line['bbox'])):
                    candidates.append(merged_line)

                skip_next_line = True
                continue

        if is_large and looks_like_title:
            candidates.append(ln)
            
    return candidates

def _merge_connected_titles(titles) -> list:
    """
    Post-processes the extracted titles to:
    1. Merge section numbers (e.g. '1.') with their following title line.
    2. Merge titles that are very close vertically (less than 16 units apart) and on the same page.
    """
    merged_titles = []
    i = 0
    while i < len(titles):
        current_title = titles[i]

        # Check if the current title is a section number (e.g., "1." or "1.2.")
        if re.match(r'^\d+(\.\d+)*\.$', current_title['text']) and (i + 1) < len(titles):
            next_title = titles[i + 1]
            # Merge the current title (section number) with the next one
            merged_text = f"{current_title['text']} {next_title['text']}"
            merged_bbox = [
                min(current_title['bbox'][0], next_title['bbox'][0]),
                min(current_title['bbox'][1], next_title['bbox'][1]),
                max(current_title['bbox'][2], next_title['bbox'][2]),
                max(current_title['bbox'][3], next_title['bbox'][3]),
            ]

            merged_entry = {
                'page': current_title['page'],
                'text': merged_text,
                'bbox': tuple(merged_bbox),
            }
            # Move past the next title
            i += 2
        else:
            # If not a section heading, start with the current title as a base
            merged_entry = current_title
            i += 1

        # Now, try to merge with subsequent lines that are close vertically
        # Keep merging if:
        # - The next title is on the same page
        # - The vertical distance between the merged_entry and the next title is < 16
        # - The next title is not a section heading itself (no need to merge again)
        while i < len(titles):
            next_title = titles[i]
            
            # Check if on the same page
            if next_title['page'] != merged_entry['page']:
                break

            # Calculate vertical distance
            # We'll consider the vertical gap as the difference between the top of the next line 
            # and the bottom of the current merged_entry.
            current_bottom = merged_entry['bbox'][3]
            next_top = next_title['bbox'][1]
            vertical_gap = next_top - current_bottom

            # Check if next title looks like a section heading
            is_section_heading = bool(re.match(r'^\d+(\.\d+)*\.$', next_title['text']))

            if vertical_gap < 16 and not is_section_heading:
                # Merge them
                merged_text = f"{merged_entry['text']} {next_title['text']}"
                merged_bbox = (
                    min(merged_entry['bbox'][0], next_title['bbox'][0]),
                    min(merged_entry['bbox'][1], next_title['bbox'][1]),
                    max(merged_entry['bbox'][2], next_title['bbox'][2]),
                    max(merged_entry['bbox'][3], next_title['bbox'][3]),
                )

                merged_entry = {
                    'page': merged_entry['page'],
                    'text': merged_text,
                    'bbox': merged_bbox
                }

                i += 1
            else:
                # Either not close enough or is a section heading, so stop merging
                break

        merged_titles.append(merged_entry)

    return merged_titles

def extract_titles_from_pdf(pdf_path) -> list:
    """
    Extracts titles from a PDF file, ignoring text lines that are part of tables, 
    and identifying likely title candidates based on font size, formatting, and position.
    Skips pages containing "Indice Generale".
    
    Returns:
    - A list of dictionaries where each dictionary contains:
      - 'page': Page number where the title appears
      - 'text': The extracted title text
      - 'bbox': The bounding box (x0, y0, x1, y1) of the title
    """
    with pdfplumber.open(pdf_path) as plumber_doc:
        doc = fitz.open(pdf_path)  # For detecting the position and coordinates of the text elements
        titles = []  # To store the extracted titles

        for page_number, page in enumerate(doc, start=1):
            try:
                plumber_page = plumber_doc.pages[page_number - 1]

                # Check if the page contains "Indice Generale"
                page_text = plumber_page.extract_text()
                if "Indice generale" in page_text:
                    continue  # Skip this page if it contains "Indice Generale"

                # Identify tables on the page
                tables = plumber_page.find_tables()
                table_bboxes = [t.bbox for t in tables] if tables else []

                # Extract all text lines and associated metadata from the page
                lines = _extract_page_lines(page)

                # Mark lines that intersect with tables
                _mark_lines_in_tables(lines, table_bboxes)

                # Filter out lines inside tables
                lines = [l for l in lines if not l.get('in_table', False)]
                if not lines:
                    continue

                # Sort lines by their vertical and horizontal positions
                lines.sort(key=lambda l: (l['bbox'][1], l['bbox'][0]))

                # Calculate the median font size for the current page
                font_sizes = [l['size'] for l in lines]
                median_font_size = statistics.median(font_sizes) if font_sizes else 0

                # Detect the main title on the first page
                if page_number == 1:
                    main_title_line = _find_main_title(lines, page.rect.height, page.rect.width)
                    if main_title_line:
                        clean_main_title = _clean_text(main_title_line['text'])
                        if clean_main_title:
                            titles.append({
                                'page': page_number,
                                'text': clean_main_title,
                                'bbox': main_title_line['bbox']
                            })
                    continue  # Skip further processing for the first page

                # Detect other candidate titles for pages other than the first
                candidate_lines = _find_title_candidates(lines, median_font_size, page_number)

                # De-duplicate and clean candidate titles
                unique_candidates = {(cl['text'], tuple(cl['bbox'])): cl for cl in candidate_lines}
                filtered_candidates = []
                for (text, bbox), cl in unique_candidates.items():
                    clean_text_val = _clean_text(text)
                    if clean_text_val:
                        filtered_candidates.append({
                            'page': page_number,
                            'text': clean_text_val,
                            'bbox': cl['bbox']
                        })

                # Sort candidates by position on the page
                filtered_candidates.sort(key=lambda c: (c['bbox'][1], c['bbox'][0]))
                titles.extend(filtered_candidates)
            except Exception as e:
                print(f"Error processing page {page_number}: {e}")

        # Sort titles by page number and their positions on the page
        titles.sort(key=lambda t: (t['page'], t['bbox'][1], t['bbox'][0]))

        # Post-process titles to merge connected lines
        titles = _merge_connected_titles(titles)

        return titles

# --- Main Processing ---

def main():
    """Main function to orchestrate the processing of PDF files."""
    align_working_directory()
    pdf_files = find_files_in_data_directory("pdf")

    if not pdf_files:
        print("No PDF files found to process.")
        return

    for pdf_path in pdf_files:
        print(f"\nProcessing PDF: {pdf_path}\n")
        titles = extract_titles_from_pdf(pdf_path)
        if titles:
            for title in titles:
                print(f"Page {title['page']}: {title['text']}")
        else:
            print(f"No titles found in {pdf_path}")


def align_working_directory():
    """Aligns the working directory with the script's directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)


def find_files_in_data_directory(extension) -> list:
    """Finds files with the given extension in the 'data' directory."""
    return glob.glob(os.path.join(".", "data", f"*.{extension}"))


if __name__ == "__main__":
    main()