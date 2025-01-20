# extractor_api/apps.py

from django.apps import AppConfig
import logging

class ExtractorAPIConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'extractor_api'

    def ready(self):
        from . import services
        try:
            services.init_app()
            logging.getLogger(__name__).info("Extractor API services initialized successfully.")
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to initialize Extractor API services: {e}")
