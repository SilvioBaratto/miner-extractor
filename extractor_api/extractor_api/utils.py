import pdfplumber
from typing import List, Dict, Optional, Any
import math
import re
import pandas as pd
from collections import defaultdict
import fitz
from urllib.parse import urlparse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def round_or_ceil(value: float) -> int:
    return math.floor(value)

def _group_words_into_lines(words, vertical_tolerance=3):
    words = sorted(words, key=lambda w: (round(w['top']), w['x0']))

    lines = []
    current_line = {
        'text': [],
        'x0': None,
        'x1': None,
        'top': None,
        'bottom': None
    }

    for w in words:
        if current_line['top'] is None:
            # Start a new line
            current_line['top'] = w['top']
            current_line['bottom'] = w['bottom']
            current_line['x0'] = w['x0']
            current_line['x1'] = w['x1']
            current_line['text'].append(w['text'])
        else:
            # Check if the current word is on the same line
            if abs(w['top'] - current_line['top']) <= vertical_tolerance:
                # Same line
                current_line['text'].append(w['text'])
                current_line['x1'] = max(current_line['x1'], w['x1'])
                current_line['bottom'] = max(current_line['bottom'], w['bottom'])
            else:
                # Finish current line and start a new one
                lines.append({
                    'text': ' '.join(current_line['text']),
                    'x0': current_line['x0'],
                    'x1': current_line['x1'],
                    'top': current_line['top'],
                    'bottom': current_line['bottom']
                })
                current_line = {
                    'text': [w['text']],
                    'x0': w['x0'],
                    'x1': w['x1'],
                    'top': w['top'],
                    'bottom': w['bottom']
                }

    # Append the last line if it exists
    if current_line['text']:
        lines.append({
            'text': ' '.join(current_line['text']),
            'x0': current_line['x0'],
            'x1': current_line['x1'],
            'top': current_line['top'],
            'bottom': current_line['bottom']
        })

    return lines

def _extract_table_bboxes(page):
    table_bboxes = []
    try:
        tables = page.find_tables()
        for tbl in tables:
            table_bboxes.append(tbl.bbox)  # (x0, top, x1, bottom)
    except Exception as e:
        print(f"Error extracting table bboxes on page {page.page_number}: {e}")
    return table_bboxes

def _is_line_in_table(line_bbox, table_bboxes):
    for table_bbox in table_bboxes:
        tx0, ttop, tx1, tbottom = table_bbox
        lx0, ltop, lx1, lbottom = line_bbox
        # Check overlap
        if not (lx1 < tx0 or lx0 > tx1) and not (lbottom < ttop or ltop > tbottom):
            return True
    return False

def _collect_all_lines(pdf_path):
    all_pages_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words()
            if not words:
                all_pages_data.append({'page_number': page_number, 'lines': []})
                continue

            lines = _group_words_into_lines(words)
            table_bboxes = _extract_table_bboxes(page)

            filtered_lines = []
            for line in lines:
                line_bbox = (line['x0'], line['top'], line['x1'], line['bottom'])
                if _is_line_in_table(line_bbox, table_bboxes):
                    continue

                cleaned_text = _clean_text(line['text'])
                if not cleaned_text:
                    continue

                if _is_unwanted_line(cleaned_text):
                    continue

                # Optional header/footer removal
                page_height = page.height
                top_margin = 50  
                bottom_margin = 50

                if line['top'] < top_margin:
                    continue  # Likely a header
                if (page_height - line['bottom']) < bottom_margin:
                    continue  # Likely a footer

                filtered_lines.append({
                    'text': cleaned_text,
                    'x0': line['x0'],
                    'x1': line['x1'],
                    'top': line['top'],
                    'bottom': line['bottom']
                })

            all_pages_data.append({
                'page_number': page_number,
                'lines': filtered_lines,
            })

    return all_pages_data

def _clean_text(text: str) -> str:
    patterns = [
        # Remove unwanted characters
    ]

    cleaned_text = text
    for pattern in patterns:
        cleaned_text = pattern.sub("", cleaned_text)
    
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def _is_unwanted_line(text: str) -> bool:
    unwanted_keywords = [
        # Add more keywords as needed
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in unwanted_keywords)

def _extract_links_from_table(
    table: pdfplumber.table.Table,
    words: List[Dict[str, Any]],
    hyperlinks: List[Dict[str, Any]],
    url_pattern: re.Pattern
) -> List[Dict]:
    row_tolerance = 5
    sorted_cells = sorted(table.cells, key=lambda c: (c[1], c[0]))

    rows = []
    current_row = []
    current_top = None
    for cell in sorted_cells:
        x0, top, x1, bottom = cell
        if current_top is None:
            current_top = top
            current_row = [cell]
        elif abs(top - current_top) <= row_tolerance:
            current_row.append(cell)
        else:
            rows.append(current_row)
            current_row = [cell]
            current_top = top
    if current_row:
        rows.append(current_row)

    link_records = []
    for row_idx, row_cells in enumerate(rows, start=1):
        row_cells_sorted = sorted(row_cells, key=lambda c: c[0])
        for col_idx, cell in enumerate(row_cells_sorted, start=1):
            x0, top, x1, bottom = cell
            cell_words = [
                w for w in words
                if (w["x0"] >= x0 and w["x1"] <= x1
                    and w["top"] >= top and w["bottom"] <= bottom)
            ]
            cell_text = ' '.join(w["text"] for w in cell_words)

            # 1) Regex-based link detection
            url_matches = url_pattern.findall(cell_text)
            for match in url_matches:
                link_records.append(match)

            # 2) Annotation-based link detection
            for link_annot in hyperlinks:
                link_uri = link_annot.get("uri")
                if not link_uri:
                    continue
                lx0, ltop, lx1, lbottom = link_annot["x0"], link_annot["top"], link_annot["x1"], link_annot["bottom"]
                # Overlap check
                if not (lx1 < x0 or lx0 > x1 or lbottom < top or ltop > bottom):
                    link_records.append(link_uri)

    return link_records

def _remove_text_from_page(page: pdfplumber.page.Page, cleaned_text: str) -> pdfplumber.page.Page:
    # Stub—no actual text removal in pdfplumber
    return page

def _merge_table(
    page_num: int, 
    table: pdfplumber.table.Table, 
    prev_page_num: Optional[int], 
    prev_table: Optional[Dict[str, Any]], 
    page_content: pdfplumber.page.Page,
    text_between: bool
):
    """
    Merges tables if:
      - There's a 'prev_table'
      - There's NO 'text_between'
      - The bounding-box x-range is the same
      - The previous page is exactly (current page - 1)
    Also handles link extraction for each table chunk.
    """
    try:
        data_table = table.extract()
        if not data_table:
            return pd.DataFrame(), False, prev_table

        url_pattern = re.compile(r'https?://\S+')
        words = page_content.extract_words()
        hyperlinks = page_content.hyperlinks or []
        table_links = _extract_links_from_table(table, words, hyperlinks, url_pattern)

        table_bbox_0 = round_or_ceil(table.bbox[0])
        table_bbox_2 = round_or_ceil(table.bbox[2])

        merged_from_previous = False

        if prev_table and not text_between:
            prev_table_bbox_0 = round_or_ceil(prev_table["bbox"][0])
            if (
                prev_page_num is not None
                and prev_page_num == (page_num - 1)
                and prev_table_bbox_0 == table_bbox_0
            ):
                # Merge: combine data and links from the previous table
                data_table = prev_table["data"] + data_table
                table_links = prev_table["links"] + table_links
                merged_from_previous = True

        if merged_from_previous:
            # Instead of inheriting the old page_start,
            # update it to the current page_num where the table links are found.
            page_start = page_num
            new_bbox = (
                table_bbox_0,
                min(prev_table["bbox"][1], table.bbox[1]),
                table_bbox_2,
                max(prev_table["bbox"][3], table.bbox[3])
            )
            merged_table = {
                "data": data_table,
                "bbox": new_bbox,
                "links": table_links,
                "page_start": page_start  # Updated to current page
            }
            df = pd.DataFrame(data_table)
            df = df.replace('\n', '  ', regex=True)
            return df, True, merged_table
        else:
            # For a brand new table, use the current page number.
            current_table = {
                "data": data_table,
                "bbox": table.bbox,
                "links": table_links,
                "page_start": page_num
            }
            df = pd.DataFrame(data_table)
            df = df.replace('\n', '  ', regex=True)
            return df, False, current_table

    except Exception as e:
        raise Exception(f"Error in merging tables: {e}")

def _extract_all_page_links(pdf_path: str) -> Dict[str, List[str]]:
    """
    Returns a dict of {page_number_as_str: [list_of_links]} for the entire PDF,
    ignoring those that might appear inside tables. We'll handle that filtering later.
    """
    from urllib.parse import urlparse
    exclude_schemes = {"mailto"}  # example: skip mailto: links

    page_links_dict = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            annots = page.annots or []
            # If you also want text-based link detection, you can do a regex pass
            # on the entire page text, though you already do so in your table code.

            page_links = []
            for annot in annots:
                uri = annot.get("uri")
                if uri:
                    parsed = urlparse(uri)
                    if parsed.scheme.lower() in exclude_schemes:
                        # skip excluded links
                        continue
                    page_links.append(uri)

            # Put the final page-links into the dictionary under their string page number
            if page_links:
                page_links_dict[str(page_idx)] = page_links

    return page_links_dict

def remove_duplicate_page_links(final_list):
    """
    Ensures that:
    - `page_links` do not contain duplicates within the same page.
    - `page_links` do not contain links already in `table_links` (on any table of the same or previous pages).
    - `page_links` are unique across pages (no duplicates in future pages).
    """
    used_table_links = set()  # Tracks all links used in table_links across all past pages
    assigned_page_links = set()  # Tracks page_links already assigned globally
    page_table_links = defaultdict(set)  # Tracks all table_links per page

    # Step 1: Collect all table_links across all pages (past & current)
    for entry in final_list:
        page_num = entry["page"]
        page_table_links[page_num].update(entry["table_links"])
        used_table_links.update(entry["table_links"])  # Store for cross-page checking

    # Step 2: Filter page_links ensuring no duplicates in past or current pages
    for entry in final_list:
        page_num = entry["page"]

        # Remove links that are in:
        # 1. Any table_links from the same page
        # 2. Any table_links from previous pages
        # 3. Any page_links already assigned globally
        filtered_links = [
            link for link in entry["page_links"]
            if link not in used_table_links  # Remove from ALL past table_links
            and link not in assigned_page_links  # Ensure no duplicates across pages
        ]

        # Store assigned page_links to prevent duplicates across pages
        assigned_page_links.update(filtered_links)

        # Assign the cleaned-up page_links
        entry["page_links"] = filtered_links

    return final_list

def extract_hyperlinks(pdf_path: str) -> List[Dict]:
    """
    Processes tables in the PDF file, merging them based on the presence of intervening text.
    Then returns a list of dicts, each describing one table's links:
      [
          {
              "page": <page_number>,
              "table_number": <table_sequence_number_on_that_page>,
              "table_links": [...],
              "page_links": [...]  # We'll fill this in at the end
          },
          ...
      ]
    """
    all_pages_data = _collect_all_lines(pdf_path)
    pages_dict: Dict[str, Dict[str, Any]] = {}
    prev_table = None
    prev_page_num = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            cleaned_text = _clean_text(page_text) if page_text else ""
            page_content = _remove_text_from_page(page, cleaned_text)
            tables = page_content.find_tables()

            lines = (
                all_pages_data[page_num - 1]['lines']
                if (page_num - 1) < len(all_pages_data) else []
            )

            if not tables:
                continue

            # Sort top->bottom
            sorted_tables = sorted(tables, key=lambda tbl: tbl.bbox[1])

            for tbl in sorted_tables:
                try:
                    # text_between could be True if you implement that logic
                    text_between = False
                    df, merged_from_previous, new_prev_table = _merge_table(
                        page_num,
                        tbl,
                        prev_page_num,
                        prev_table,
                        page_content,
                        text_between
                    )

                    prev_table = new_prev_table
                    prev_page_num = page_num

                    # Identify the "start_page" for the (merged) table
                    start_page = str(new_prev_table["page_start"])  
                    links_for_this_table = new_prev_table["links"]

                    # Make sure the dictionary for this start_page exists
                    if start_page not in pages_dict:
                        pages_dict[start_page] = {"_counter": 0}

                    if merged_from_previous:
                        # Remove the last table we inserted for this page
                        last_label = f"table_{pages_dict[start_page]['_counter']}"
                        pages_dict[start_page].pop(last_label, None)
                        # DO NOT increment the counter here
                    else:
                        # Brand new table chunk => increment the counter
                        pages_dict[start_page]["_counter"] += 1

                    # Now reuse the existing or newly incremented counter
                    table_label = f"table_{pages_dict[start_page]['_counter']}"
                    pages_dict[start_page][table_label] = links_for_this_table

                except Exception as e:
                    print(f"Error processing table on page {page_num}: {e}")

    # 2) Remove the "_counter" from each page's dictionary
    for pg in pages_dict:
        pages_dict[pg].pop("_counter", None)

    # 3) Convert that dictionary to the list-of-dicts structure you want
    final_list: List[Dict] = []
    for page_str, table_info in pages_dict.items():
        page_int = int(page_str)  # convert "7" -> 7
        # table_info is a dict like { "table_1": [...], "table_2": [...] }
        for tbl_key, links_list in table_info.items():
            if tbl_key.startswith("table_"):
                table_num = int(tbl_key.split("_")[1])
                final_list.append({
                    "page": page_int,
                    "table_number": table_num,
                    "table_links": links_list,
                    "page_links": []  # We will fill this in a final step
                })

    # (Optional) Sort the final list by page, then table_number
    final_list.sort(key=lambda x: (x["page"], x["table_number"]))
    
    # 4) Gather page-level links (outside tables) using a separate function.
    page_links_dict = _extract_all_page_links(pdf_path)

    # Step 1: Ensure every page in `page_links_dict` exists in `final_list`
    existing_pages = {entry["page"] for entry in final_list}
    
    for page in page_links_dict:
        page_int = int(page)  # Convert string keys to int
        if page_int not in existing_pages:
            # If a page has no tables in `final_list`, add a placeholder entry
            final_list.append({
                "page": page_int,
                "table_number": 0,  # Placeholder value
                "table_links": [],
                "page_links": page_links_dict[page]
            })

    # Step 2: Assign page_links from `page_links_dict`
    for entry in final_list:
        page_str = str(entry["page"])
        if page_str in page_links_dict:
            # Filter out links that are already in table_links for the same entry
            entry["page_links"] = [
                link for link in page_links_dict[page_str]
                if link not in entry["table_links"]
            ]

    # Step 3: Run cleanup function to remove duplicates across pages
    final_list = remove_duplicate_page_links(final_list)

    # Step 4: Sort the final list by page and table_number
    final_list.sort(key=lambda x: (x["page"], x["table_number"]))

    # ----------------------------------------------------------------
    # 5) NOW add your embedding logic for each table
    # ----------------------------------------------------------------
    # Initialize your embedding model
    embedding_llm = HuggingFaceEmbedding(
        model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
        trust_remote_code=True
    )

    # Loop through each table entry and create embeddings for table_links and page_links
    for entry in final_list:
        # 1) Embeddings for table_links
        if entry["table_links"]:
            # Concatenate all table_links into a single string
            text_to_embed = " ".join(str(link) for link in entry["table_links"])
            
            # HuggingFaceEmbeddings.embed_documents expects a list of documents.
            # We'll pass just one combined text, and then take the first embedding result.
            embeddings_list = embedding_llm.get_text_embedding(text_to_embed)
            entry["tbl_embed_vector"] = embeddings_list
        else:
            entry["tbl_embed_vector"] = []

        # 2) Embeddings for page_links
        if entry["page_links"]:
            # Concatenate all page_links into a single string
            text_to_embed = " ".join(str(link) for link in entry["page_links"])
            
            embeddings_list = embedding_llm.get_text_embedding(text_to_embed)
            entry["pg_embed_vector"] = embeddings_list
        else:
            entry["pg_embed_vector"] = []

    # Finally, return the list
    return final_list

def compute_cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    """
    Compute the cosine similarity between two vectors.
    """
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def merge_markdown_and_tables(
    sections: list[dict],
    table_data: list[dict],
) -> list[dict]:
    """
    For each Markdown 'section':
      - If it contains a <table> tag, find the best-matching table (via cosine similarity).
        Then attach that table's `table_links` AND `page_links` at the end of the content.
      - If NO <table> is found, similarly find the best page-level match (via the `pg_embed_vector`)
        and attach that entry's `page_links`.

    Args:
        sections (list[dict]): Each section with {'content': str, 'page_idx': int}.
        table_data (list[dict]): List of tables with metadata, links, and embeddings.

    Returns:
        list[dict]: Updated sections with embedded table and page links at the end of the content.
    """
    embedding_llm = HuggingFaceEmbedding(
        model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
        trust_remote_code=True
    )

    for sec in sections:
        sec["table_links"] = []
        sec["page_links"] = []
        content_text = sec.get("content", "")
        page_idx = sec.get("page_idx", -1)

        # If no text, skip it
        if not content_text:
            continue

        # Gather only the table_data entries that belong to this page
        page_entries = [tbl for tbl in table_data if tbl.get("page") == page_idx]

        # Check for <table> tags
        table_tags = re.findall(r'(?i)<table', content_text)

        if not table_tags:
            #
            # 1) If there's NO <table> tag, try to find the best chunk match
            #    among the page_entries based on `pg_embed_vector`.
            #
            if not page_entries:
                # No table_data for this page at all
                continue

            # Embed this chunk
            chunk_embedding = embedding_llm.get_text_embedding(content_text)

            best_page_entry = None
            best_score = -1.0
            for entry in page_entries:
                pg_emb = entry.get("pg_embed_vector", [])
                if pg_emb:
                    sim = compute_cosine_similarity(chunk_embedding, pg_emb)
                    if sim > best_score and sim >= 0.4:
                        best_page_entry = entry
                        best_score = sim

            if best_page_entry:
                # We found a good page-level match
                sec["page_links"] = best_page_entry.get("page_links", [])
            else:
                # If you prefer no links if below threshold:
                sec["page_links"] = []
                #
                # Or, if you still want to attach all page links when
                # none meets the similarity threshold, uncomment below:
                # sec["page_links"] = list({
                #     plink
                #     for tbl in page_entries
                #     for plink in tbl.get("page_links", [])
                # })
        else:
            #
            # 2) If there IS a <table> tag, keep your existing table logic as-is.
            #
            chunk_embedding = embedding_llm.get_text_embedding(content_text)
            best_table, best_table_score = None, -1.0

            # Find the best table match across *all* table_data
            for tbl in table_data:
                tbl_vector = tbl.get("tbl_embed_vector", [])
                if tbl_vector:
                    similarity = compute_cosine_similarity(chunk_embedding, tbl_vector)
                    if similarity > best_table_score and similarity >= 0.5:
                        best_table = tbl
                        best_table_score = similarity

            if best_table:
                # Attach both table-level and page-level links for the matched table
                sec["table_links"] = best_table.get("table_links", [])
                sec["page_links"] = best_table.get("page_links", [])
            else:
                # If no good table match, you can still do a best page link match
                # or leave it as you originally do. Here’s a minimal fallback:
                all_page_links = list({
                    plink
                    for tbl in page_entries
                    for plink in tbl.get("page_links", [])
                })
                sec["page_links"] = all_page_links

    return sections

def main():
    pdf_path = "extractor_api/media/extracted/IS-01 Elenco delle applicazioni informatiche validate in sicurezza_layout.pdf"
    final_data = extract_hyperlinks(pdf_path)

    import json
    print(json.dumps(final_data, indent=2))

    with open('table_data.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
