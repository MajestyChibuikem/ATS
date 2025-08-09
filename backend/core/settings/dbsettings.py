"""
Database settings for SSAS (Smart Student Analytics System).

This file contains database-specific configurations and optimizations.
"""

# Database connection settings
DB_OPTIONS = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'ssas_db',
        'USER': 'ssas_user',
        'PASSWORD': '',
        'HOST': 'localhost',
        'PORT': '5432',
        'OPTIONS': {
            'sslmode': 'prefer',
            'connect_timeout': 10,
        },
        'CONN_MAX_AGE': 60,  # Connection pooling
        'ATOMIC_REQUESTS': True,  # Wrap each request in a transaction
        'OPTIONS': {
            'sslmode': 'prefer',
            'connect_timeout': 10,
            'application_name': 'ssas_backend',
        },
    }
}

# Database optimization settings
DB_OPTIMIZATIONS = {
    'ENABLE_QUERY_LOGGING': False,  # Set to True for debugging
    'SLOW_QUERY_THRESHOLD': 1.0,  # Log queries taking more than 1 second
    'MAX_CONNECTIONS': 20,
    'MIN_CONNECTIONS': 5,
    'CONNECTION_TIMEOUT': 10,
    'IDLE_TIMEOUT': 300,  # 5 minutes
}

# Database backup settings
DB_BACKUP_SETTINGS = {
    'ENABLE_AUTO_BACKUP': True,
    'BACKUP_FREQUENCY': 'daily',  # daily, weekly, monthly
    'BACKUP_RETENTION_DAYS': 30,
    'BACKUP_PATH': '/backups/ssas/',
    'COMPRESS_BACKUPS': True,
}

# Database migration settings
DB_MIGRATION_SETTINGS = {
    'AUTO_MIGRATE': True,
    'MIGRATION_TIMEOUT': 300,  # 5 minutes
    'ROLLBACK_ON_FAILURE': True,
}

# Database performance monitoring
DB_MONITORING = {
    'ENABLE_PERFORMANCE_MONITORING': True,
    'LOG_SLOW_QUERIES': True,
    'SLOW_QUERY_THRESHOLD_MS': 1000,
    'LOG_CONNECTION_POOL_STATS': True,
    'LOG_DEADLOCKS': True,
} 