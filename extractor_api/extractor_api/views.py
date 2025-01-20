import os
from django.http import JsonResponse
from rest_framework.decorators import (
    api_view, 
    permission_classes, 
    authentication_classes,
    parser_classes
)
from rest_framework.permissions import AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.conf import settings
from .services import extract_pdf_content, parse_markdown_with_pages

pdf_file_param = openapi.Parameter(
    name='file',
    in_=openapi.IN_FORM,
    description="PDF file to extract content from",
    type=openapi.TYPE_FILE,
    required=True
)

@swagger_auto_schema(
    method='post',
    manual_parameters=[pdf_file_param],
    consumes=['multipart/form-data'],
    responses={200: openapi.Response('Markdown content as JSON', schema=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'content': openapi.Schema(type=openapi.TYPE_STRING, description='Extracted markdown content'),
        }
    ))},
)
@api_view(['POST'])
@permission_classes([AllowAny])
@authentication_classes([])
@parser_classes([MultiPartParser, FormParser])
def extract_pdf(request):
    """API endpoint to handle PDF extraction and return the Markdown content in JSON."""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        pdf_file = request.FILES['file']
        output_directory = os.path.join(settings.MEDIA_ROOT, 'extracted')
        os.makedirs(output_directory, exist_ok=True)

        input_pdf_path = os.path.join(output_directory, pdf_file.name)
        with open(input_pdf_path, 'wb') as f:
            for chunk in pdf_file.chunks():
                f.write(chunk)

        markdown_file_path = extract_pdf_content(input_pdf_path, output_directory)

        if not os.path.exists(markdown_file_path):
            return JsonResponse({"error": "Extraction failed"}, status=500)

        with open(markdown_file_path, 'r', encoding='utf-8') as md_file:
            content = md_file.read()

        return JsonResponse({"content": content})

    except Exception as e:
        return JsonResponse({"error": "An error occurred", "details": str(e)}, status=500)

@swagger_auto_schema(
    method='post',
    manual_parameters=[pdf_file_param],
    consumes=['multipart/form-data'],
    responses={200: openapi.Response('Extracted structured JSON with page numbers', schema=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'sections': openapi.Schema(
                type=openapi.TYPE_ARRAY,
                items=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'title': openapi.Schema(type=openapi.TYPE_STRING, description='Section title'),
                        'content': openapi.Schema(type=openapi.TYPE_STRING, description='Section content'),
                        'page_number': openapi.Schema(type=openapi.TYPE_INTEGER, description='Page number')
                    }
                )
            )
        }
    ))},
)
@api_view(['POST'])
@permission_classes([AllowAny])
@authentication_classes([])
@parser_classes([MultiPartParser, FormParser])
def extract_to_qdrant(request):
    """API endpoint to parse extracted markdown and return structured JSON with page numbers."""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        pdf_file = request.FILES['file']
        output_directory = os.path.join(settings.MEDIA_ROOT, 'extracted')
        os.makedirs(output_directory, exist_ok=True)

        input_pdf_path = os.path.join(output_directory, pdf_file.name)
        with open(input_pdf_path, 'wb') as f:
            for chunk in pdf_file.chunks():
                f.write(chunk)

        markdown_file_path, content_list_path = extract_pdf_content(input_pdf_path, output_directory)

        if not os.path.exists(markdown_file_path) or not os.path.exists(content_list_path):
            return JsonResponse({"error": "Extraction failed"}, status=500)

        with open(markdown_file_path, 'r', encoding='utf-8') as md_file:
            markdown_content = md_file.read()

        parsed_sections = parse_markdown_with_pages(markdown_content, content_list_path)

        return JsonResponse({"sections": parsed_sections})

    except Exception as e:
        return JsonResponse({"error": "An error occurred", "details": str(e)}, status=500)