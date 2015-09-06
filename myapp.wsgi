import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, '/var/www/gradranker.com/public_html')

from myapp import app as application
