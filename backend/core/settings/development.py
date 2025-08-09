"""
Development settings for SSAS (Smart Student Analytics System).

This file contains settings specific to the development environment.
"""

from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1',
    '0.0.0.0',
]

# Database
# https://docs.djangoproject.com/en/5.1/ref/settings/#databases
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Email settings for development
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# Debug toolbar settings (if needed)
if DEBUG:
    INSTALLED_APPS += ['debug_toolbar']
    MIDDLEWARE += ['debug_toolbar.middleware.DebugToolbarMiddleware']
    INTERNAL_IPS = [
        '127.0.0.1',
        'localhost',
    ]

# Development-specific logging
LOGGING['loggers']['core']['level'] = 'DEBUG'

# CORS settings for development
CORS_ALLOW_ALL_ORIGINS = True  # Only for development!

# Cache settings for development
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
        'TIMEOUT': 300,
        'OPTIONS': {
            'MAX_ENTRIES': 1000,
        }
    }
}

# Static files for development
STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.StaticFilesStorage'

# Media files for development
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Development-specific ML settings
ML_MODEL_SETTINGS.update({
    'DEBUG_MODE': True,
    'MODEL_SAVE_PATH': BASE_DIR / 'ml_models' / 'development',
    'LOG_PREDICTIONS': True,
    'SAVE_TRAINING_LOGS': True,
})

# Analytics settings for development
ANALYTICS_SETTINGS.update({
    'ENABLE_DEBUG_LOGGING': True,
    'SAVE_ANALYTICS_LOGS': True,
    'CACHE_TIMEOUT': 60,  # 1 minute for faster development
})

# Create necessary directories
import os
os.makedirs(BASE_DIR / 'ml_models' / 'development', exist_ok=True)
os.makedirs(BASE_DIR / 'media', exist_ok=True)
os.makedirs(BASE_DIR / 'static', exist_ok=True) 