"""gunicorn settings for the wine API container.

Cloud Run tells the container which port to listen on through the PORT environment variable, so the bind address is read from it (8080 locally).
"""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"
workers = 1      
threads = 8      
timeout = 0     