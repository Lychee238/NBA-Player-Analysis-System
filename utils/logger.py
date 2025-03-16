import logging
import os
import sys
import codecs
from logging.handlers import RotatingFileHandler
from flask import has_request_context, request
from pathlib import Path

class RequestFormatter(logging.Formatter):
    """Custom log formatter that adds request-related information"""
    
    def format(self, record):
        if has_request_context():
            record.url = request.url
            record.remote_addr = request.remote_addr
            record.method = request.method
            record.path = request.path
        else:
            record.url = None
            record.remote_addr = None
            record.method = None
            record.path = None
            
        # Ensure message is string type
        if isinstance(record.msg, bytes):
            record.msg = record.msg.decode('utf-8')
        elif not isinstance(record.msg, str):
            record.msg = str(record.msg)
            
        return super().format(record)

class UTF8FileHandler(RotatingFileHandler):
    """File handler that supports UTF-8-SIG (with BOM) encoding"""
    
    def _open(self):
        if not os.path.exists(self.baseFilename):
            # Create file with BOM if it doesn't exist
            with codecs.open(self.baseFilename, 'w', encoding='utf-8-sig') as f:
                pass
        return codecs.open(self.baseFilename, self.mode, encoding='utf-8-sig')

def init_app(app):
    """Initialize the application's logging system"""
    
    # Ensure log directory exists
    log_dir = Path('logs')
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    
    # Set default encoding to UTF-8
    if sys.platform.startswith('win'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    
    # Create custom formatter
    log_format = (
        '[%(asctime)s] %(remote_addr)s %(method)s %(url)s\n'
        '%(levelname)s in %(module)s: %(message)s\n'
        '----------------------------------------'
    )
    formatter = RequestFormatter(log_format)
    
    # Configure file handler with custom UTF8FileHandler
    file_handler = UTF8FileHandler(
        'logs/app.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10,
        mode='a'  # Append mode
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    
    # Remove default handlers
    app.logger.handlers = []
    
    # Add new handlers
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    
    # Set log level
    app.logger.setLevel(logging.DEBUG if app.debug else logging.INFO)
    
    # Log application startup
    app.logger.info('Application started')
    
    return app.logger 