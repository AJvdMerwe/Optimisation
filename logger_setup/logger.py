import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(name)-12s] [%(levelname)-8s] %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger('optimisation')

# Create handlers
file_handler = logging.FileHandler('optimisation.log')
file_handler.setLevel(logging.WARNING)

# Create formatters and add it to handlers
file_format = logging.Formatter('%(asctime)s [%(name)-12s] [%(levelname)-8s] %(message)s', datefmt='%d-%b-%y %H:%M:%S')
file_handler.setFormatter(file_format)

# Add handlers to the logger
logger.addHandler(file_handler)
