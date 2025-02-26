# extractor_api/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('extract/', views.extract_pdf, name='extract_pdf'),
    path('extract_to_qdrant/', views.extract_to_qdrant, name='extract_to_qdrant'),
    path('extract_folder_to_qdrant/', views.extract_folder_to_qdrant, name='extract_folder_to_qdrant'),
]