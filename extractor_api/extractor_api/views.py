import os
import re
import glob
import json
import logging
from django.http import JsonResponse
from django.conf import settings
from rest_framework.decorators import (
    api_view, 
    permission_classes, 
    authentication_classes,
    parser_classes
)
from rest_framework.permissions import AllowAny
from rest_framework.parsers import JSONParser, FormParser, MultiPartParser
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from markdownify import markdownify as md

# Import your service functions for content extraction.
from .services import extract_pdf_content, parse_markdown_with_pages
from .utils import extract_hyperlinks, merge_markdown_and_tables

# Import Qdrant indexing utilities.
from .indexing import index_pdf, initialize_qdrant_client
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

# Define a Swagger parameter for the file upload.
pdf_file_param = openapi.Parameter(
    name='file',
    in_=openapi.IN_FORM,
    description="PDF file to extract content from",
    type=openapi.TYPE_FILE,
    required=True
)

# Swagger schema for the folder-based extraction endpoint.
folder_request_schema = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    properties={
        "folder_path": openapi.Schema(
            type=openapi.TYPE_STRING, 
            description=("Relative path from the server's explore folder that contains PDF files. "
                         "For example, 'subfolder1' or '2025/reports'.")
        ),
        "method": openapi.Schema(
            type=openapi.TYPE_STRING, 
            default="txt", 
            description="Extraction method (e.g., 'txt', 'ocr')."
        ),
        "lang": openapi.Schema(
            type=openapi.TYPE_STRING, 
            default="en", 
            description="Language for OCR (if applicable)."
        )
    },
    required=["folder_path"]
)

def save_uploaded_file(uploaded_file, output_directory):
    """
    Save the uploaded file to the output directory.
    
    Returns the full path to the saved file.
    """
    os.makedirs(output_directory, exist_ok=True)
    file_path = os.path.join(output_directory, uploaded_file.name)
    with open(file_path, 'wb') as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)
    logger.info(f"File saved to {file_path}")
    return file_path

@swagger_auto_schema(
    method='post',
    manual_parameters=[pdf_file_param],
    consumes=['multipart/form-data'],
    responses={200: openapi.Response(
        'Extracted Markdown content', 
        schema=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'content': openapi.Schema(
                    type=openapi.TYPE_STRING, 
                    description='Extracted markdown content'
                )
            }
        )
    )},
)
@api_view(['POST'])
@permission_classes([AllowAny])
@authentication_classes([])
@parser_classes([MultiPartParser, FormParser])
def extract_pdf(request):
    """
    API endpoint to extract PDF content and return the Markdown as JSON.
    The extraction method and language can be optionally provided.
    """
    try:
        # Validate file upload.
        if 'file' not in request.FILES:
            logger.error("No file uploaded")
            return JsonResponse({"error": "No file uploaded"}, status=400)
        
        pdf_file = request.FILES['file']
        
        # Allow overriding the extraction method and language via request data.
        extraction_method = request.data.get('method', 'txt')
        lang = request.data.get('lang', 'en')
        
        # Set up output directory.
        output_directory = os.path.join(settings.MEDIA_ROOT, 'extracted')
        input_pdf_path = save_uploaded_file(pdf_file, output_directory)
        
        # Extract the PDF content.
        markdown_file_path, _, _ = extract_pdf_content(
            input_pdf_path, 
            output_directory, 
            method=extraction_method, 
            lang=lang
        )
        
        if not os.path.exists(markdown_file_path):
            logger.error("Extraction failed: Markdown file not found")
            return JsonResponse({"error": "Extraction failed"}, status=500)
        
        # Read the Markdown content.
        with open(markdown_file_path, 'r', encoding='utf-8') as md_file:
            content = md_file.read()
        
        return JsonResponse({"content": content})
    
    except Exception as e:
        logger.exception("An error occurred during PDF extraction")
        return JsonResponse({"error": "An error occurred", "details": str(e)}, status=500)

@swagger_auto_schema(
    method='post',
    manual_parameters=[pdf_file_param],
    consumes=['multipart/form-data'],
    responses={200: openapi.Response(
        'Extracted structured content with page numbers', 
        schema=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'sections': openapi.Schema(
                    type=openapi.TYPE_ARRAY,
                    items=openapi.Schema(
                        type=openapi.TYPE_OBJECT,
                        properties={
                            'title': openapi.Schema(
                                type=openapi.TYPE_STRING, 
                                description='Section title'
                            ),
                            'content': openapi.Schema(
                                type=openapi.TYPE_STRING, 
                                description='Section content'
                            ),
                            'page_number': openapi.Schema(
                                type=openapi.TYPE_INTEGER, 
                                description='Page number'
                            )
                        }
                    )
                ),
                'json_file_path': openapi.Schema(
                    type=openapi.TYPE_STRING, 
                    description='Path to the JSON file with extracted sections'
                )
            }
        )
    )},
)
@api_view(['POST'])
@permission_classes([AllowAny])
@authentication_classes([])
@parser_classes([MultiPartParser, FormParser])
def extract_to_qdrant(request):
    """
    API endpoint to extract PDF content and return structured JSON (with page numbers)
    that can be used in downstream systems such as Qdrant.
    
    The extraction method and language are dynamic.
    """
    try:
        # Validate file upload.
        if 'file' not in request.FILES:
            logger.error("No file uploaded")
            return JsonResponse({"error": "No file uploaded"}, status=400)
        
        pdf_file = request.FILES['file']
        
        # Optional parameters.
        extraction_method = request.data.get('method', 'txt')
        lang = request.data.get('lang', 'en')
        
        # Set up output directory.
        output_directory = os.path.join(settings.MEDIA_ROOT, 'extracted')
        input_pdf_path = save_uploaded_file(pdf_file, output_directory)
        
        # Extract the PDF content.
        markdown_file_path, content_list_path, titles = extract_pdf_content(
            input_pdf_path, 
            output_directory, 
            method=extraction_method, 
            lang=lang
        )
        
        if not (os.path.exists(markdown_file_path) and os.path.exists(content_list_path)):
            logger.error("Extraction failed: Required files not found")
            return JsonResponse({"error": "Extraction failed"}, status=500)
        
        # Read the extracted markdown.
        with open(markdown_file_path, 'r', encoding='utf-8') as md_file:
            markdown_content = md_file.read()
        
        # Extract hyperlinks (or table data) from the PDF.
        table_data = extract_hyperlinks(input_pdf_path)
        
        # Parse markdown with page information.
        parsed_sections = parse_markdown_with_pages(markdown_content, content_list_path, titles)
        
        # Merge parsed sections with table data.
        merged_sections = merge_markdown_and_tables(parsed_sections, table_data)
        
        # Sort sections by page index (if available) for logical order.
        merged_sections = sorted(merged_sections, key=lambda x: x.get('page_idx') if x.get('page_idx') is not None else 0)
        
        # Convert HTML content to Markdown if necessary.
        html_pattern = re.compile(r'<.*?>')
        for section in merged_sections:
            section_content = section.get('content', '')
            if html_pattern.search(section_content):
                section['content'] = md(section_content)
        
        # Save the structured sections to a JSON file.
        json_output_path = os.path.join(
            output_directory, 
            f"{os.path.splitext(pdf_file.name)[0]}_qdrant.json"
        )
        with open(json_output_path, 'w', encoding='utf-8') as json_file:
            json.dump({"sections": merged_sections}, json_file, ensure_ascii=False, indent=4)
        logger.info(f"Structured JSON saved to {json_output_path}")
        
        return JsonResponse({
            "sections": merged_sections, 
            "json_file_path": json_output_path
        })
    
    except Exception as e:
        logger.exception("An error occurred during extraction to Qdrant")
        return JsonResponse({"error": "An error occurred", "details": str(e)}, status=500)

@swagger_auto_schema(
    method='post',
    request_body=folder_request_schema,
    responses={200: openapi.Response(
        "Folder extraction and Qdrant upload result",
        schema=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                "processed_files": openapi.Schema(
                    type=openapi.TYPE_ARRAY,
                    items=openapi.Schema(type=openapi.TYPE_STRING),
                    description="List of PDF files successfully processed for extraction."
                ),
                "uploaded_files": openapi.Schema(
                    type=openapi.TYPE_ARRAY,
                    items=openapi.Schema(type=openapi.TYPE_STRING),
                    description="List of PDF files successfully indexed in Qdrant."
                ),
                "failed_files": openapi.Schema(
                    type=openapi.TYPE_ARRAY,
                    items=openapi.Schema(type=openapi.TYPE_STRING),
                    description="List of PDF files that failed during extraction or upload."
                )
            }
        )
    )},
)
@api_view(['POST'])
@permission_classes([AllowAny])
@authentication_classes([])
@parser_classes([JSONParser, FormParser, MultiPartParser])
def extract_folder_to_qdrant(request):
    """
    Endpoint that accepts a relative folder path (from the explore folder) and processes all PDFs 
    (in that folder and its sub-folders). For each PDF:
      - Extracts its content.
      - Merges the extracted content.
      - Saves a JSON file with structured data.
      - Indexes/uploads the PDF to Qdrant.
    
    Returns a summary of processed, uploaded, and failed files.
    """
    try:
        # Get parameters from the request body.
        relative_folder = request.data.get('folder_path')
        extraction_method = request.data.get('method', 'txt')
        lang = request.data.get('lang', 'en')

        if not relative_folder:
            logger.error("No folder path provided.")
            return JsonResponse({"error": "No folder path provided"}, status=400)

        # Define the base "explore" folder (adjust as needed).
        base_explore_folder = os.path.join(settings.MEDIA_ROOT, 'explore')
        # Build the absolute folder path by joining the base and the relative path.
        full_folder_path = os.path.abspath(os.path.join(base_explore_folder, relative_folder))

        # Security check: Ensure the chosen folder is inside the allowed base folder.
        if not full_folder_path.startswith(os.path.abspath(base_explore_folder)):
            logger.error("Provided folder path is not allowed.")
            return JsonResponse({"error": "Provided folder path is not allowed"}, status=400)

        if not os.path.isdir(full_folder_path):
            logger.error("Provided folder path does not exist or is not a directory.")
            return JsonResponse({"error": "Provided folder path does not exist or is not a directory"}, status=400)

        # Recursively find all PDF files in the folder.
        pdf_files = glob.glob(os.path.join(full_folder_path, '**', '*.pdf'), recursive=True)
        if not pdf_files:
            logger.info("No PDF files found in the provided folder.")
            return JsonResponse({"error": "No PDF files found in the provided folder"}, status=404)

        processed_files = []
        uploaded_files = []
        failed_files = []

        # Use the common output directory for extracted content.
        output_directory = os.path.join(settings.MEDIA_ROOT, 'extracted')
        os.makedirs(output_directory, exist_ok=True)

        # Read Qdrant parameters from environment variables.
        QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
        QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333)) if os.getenv("QDRANT_HOST", "localhost") == "localhost" else None
        QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
        COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "test_collection")
        HUGGING_FACE_EMBEDDING_ENDPOINT = os.getenv('HUGGING_FACE_EMBEDDING_ENDPOINT')
        HUGGING_FACE_EMBEDDING_SPARSE_ENDPOINT = os.getenv('HUGGING_FACE_EMBEDDING_SPARSE_ENDPOINT')
        HUGGING_FACE_EMBEDDING_SIZE = int(os.getenv('HUGGING_FACE_EMBEDDING_SIZE', '768'))

        # Initialize the Qdrant client.
        client: QdrantClient = initialize_qdrant_client(QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY)

        # Process each PDF file.
        for pdf_file_path in pdf_files:
            try:
                logger.info(f"Processing PDF: {pdf_file_path}")

                # Extract PDF content.
                markdown_file_path, content_list_path, titles = extract_pdf_content(
                    pdf_file_path, 
                    output_directory, 
                    method=extraction_method, 
                    lang=lang
                )

                if not os.path.exists(markdown_file_path):
                    logger.error(f"Extraction failed for {pdf_file_path}")
                    failed_files.append(pdf_file_path)
                    continue

                # Read the extracted markdown.
                with open(markdown_file_path, 'r', encoding='utf-8') as md_file:
                    markdown_content = md_file.read()

                # Extract additional data (e.g., hyperlinks).
                table_data = extract_hyperlinks(pdf_file_path)

                # Parse and merge extracted content.
                parsed_sections = parse_markdown_with_pages(markdown_content, content_list_path, titles)
                merged_sections = merge_markdown_and_tables(parsed_sections, table_data)
                merged_sections = sorted(merged_sections, key=lambda x: x.get('page_idx') if x.get('page_idx') is not None else 0)

                # Convert any detected HTML in content to Markdown.
                html_pattern = re.compile(r'<.*?>')
                for section in merged_sections:
                    content = section.get('content', '')
                    if html_pattern.search(content):
                        section['content'] = md(content)

                # Save the structured JSON output for this PDF.
                json_output_path = os.path.join(
                    output_directory, 
                    f"{os.path.splitext(os.path.basename(pdf_file_path))[0]}_qdrant.json"
                )
                with open(json_output_path, 'w', encoding='utf-8') as json_file:
                    json.dump({"sections": merged_sections}, json_file, ensure_ascii=False, indent=4)
                logger.info(f"Extracted JSON saved to {json_output_path}")

                processed_files.append(pdf_file_path)

                # Upload the PDF to Qdrant.
                try:
                    pdf_nodes = index_pdf(
                        file_path=pdf_file_path,
                        client=client,
                        embed_model=HUGGING_FACE_EMBEDDING_ENDPOINT,
                        embed_sparse_model=HUGGING_FACE_EMBEDDING_SPARSE_ENDPOINT,
                        collection_name=COLLECTION_NAME,
                        size=HUGGING_FACE_EMBEDDING_SIZE
                    )
                    if pdf_nodes:
                        logger.info(f"Uploaded {pdf_file_path} to Qdrant successfully.")
                        uploaded_files.append(pdf_file_path)
                    else:
                        logger.error(f"Indexing returned no nodes for {pdf_file_path}.")
                        failed_files.append(pdf_file_path)
                except Exception as upload_exc:
                    logger.exception(f"Error uploading {pdf_file_path} to Qdrant: {upload_exc}")
                    failed_files.append(pdf_file_path)

            except Exception as proc_exc:
                logger.exception(f"Error processing file {pdf_file_path}: {proc_exc}")
                failed_files.append(pdf_file_path)

        return JsonResponse({
            "processed_files": processed_files,
            "uploaded_files": uploaded_files,
            "failed_files": failed_files
        })

    except Exception as e:
        logger.exception("An error occurred during folder extraction to Qdrant")
        return JsonResponse({"error": "An error occurred", "details": str(e)}, status=500)