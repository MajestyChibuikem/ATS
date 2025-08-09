"""
Custom error handlers for SSAS (Smart Student Analytics System).
"""

from django.shortcuts import render
from django.http import JsonResponse
from django.template import loader
from django.http import Http404
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


def handler404(request, exception):
    """
    Custom 404 error handler.
    
    Args:
        request: The HTTP request object
        exception: The exception that was raised
        
    Returns:
        HttpResponse: Custom 404 response
    """
    logger.warning(f"404 error for URL: {request.path}")
    
    if request.path.startswith('/api/'):
        # Return JSON response for API requests
        return JsonResponse({
            'error': 'Not Found',
            'message': 'The requested resource was not found.',
            'path': request.path
        }, status=404)
    
    # Return HTML response for web requests
    context = {
        'error_code': 404,
        'error_message': 'Page Not Found',
        'error_description': 'The page you are looking for does not exist.',
    }
    
    template = loader.get_template('errors/404.html')
    return template.render(context, request)


def handler500(request):
    """
    Custom 500 error handler.
    
    Args:
        request: The HTTP request object
        
    Returns:
        HttpResponse: Custom 500 response
    """
    logger.error(f"500 error for URL: {request.path}")
    
    if request.path.startswith('/api/'):
        # Return JSON response for API requests
        return JsonResponse({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred. Please try again later.',
            'path': request.path
        }, status=500)
    
    # Return HTML response for web requests
    context = {
        'error_code': 500,
        'error_message': 'Internal Server Error',
        'error_description': 'An unexpected error occurred. Please try again later.',
    }
    
    template = loader.get_template('errors/500.html')
    return template.render(context, request)


def handler403(request, exception):
    """
    Custom 403 error handler.
    
    Args:
        request: The HTTP request object
        exception: The exception that was raised
        
    Returns:
        HttpResponse: Custom 403 response
    """
    logger.warning(f"403 error for URL: {request.path}")
    
    if request.path.startswith('/api/'):
        # Return JSON response for API requests
        return JsonResponse({
            'error': 'Forbidden',
            'message': 'You do not have permission to access this resource.',
            'path': request.path
        }, status=403)
    
    # Return HTML response for web requests
    context = {
        'error_code': 403,
        'error_message': 'Access Forbidden',
        'error_description': 'You do not have permission to access this resource.',
    }
    
    template = loader.get_template('errors/403.html')
    return template.render(context, request)


def handler400(request, exception):
    """
    Custom 400 error handler.
    
    Args:
        request: The HTTP request object
        exception: The exception that was raised
        
    Returns:
        HttpResponse: Custom 400 response
    """
    logger.warning(f"400 error for URL: {request.path}")
    
    if request.path.startswith('/api/'):
        # Return JSON response for API requests
        return JsonResponse({
            'error': 'Bad Request',
            'message': 'The request could not be processed due to invalid data.',
            'path': request.path
        }, status=400)
    
    # Return HTML response for web requests
    context = {
        'error_code': 400,
        'error_message': 'Bad Request',
        'error_description': 'The request could not be processed due to invalid data.',
    }
    
    template = loader.get_template('errors/400.html')
    return template.render(context, request) 