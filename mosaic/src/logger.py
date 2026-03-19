import logging
import os


class InMemoryLogHandler(logging.Handler):
    """
    A logging handler that collects log messages in memory.
    The messages are stored in the `logs` list.
    """

    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        msg = self.format(record)
        self.logs.append(msg)


def setup_logger(name: str) -> logging.Logger:
    """
    Set up and return a logger that outputs both to the console and to a file
    (server.log on the Desktop/log folder).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console output.
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File output: prefer project-relative log dir (e.g. mosaic/log or cwd/log)
        _mosaic_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(_mosaic_root, "log")
        try:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_path = os.path.join(log_dir, "server.log")
        except OSError:
            log_dir = os.path.join(os.getcwd(), "log")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "server.log")

        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def setup_conversation_logger(conversation_id: str):
    """
    Set up and return a conversation-specific logger that outputs to a log file and
    also maintains an in-memory list of log messages.

    Returns a tuple: (logger, in_memory_log_handler)
    """
    logger_name = f"conversation.{conversation_id}"
    conv_logger = logging.getLogger(logger_name)
    conv_logger.setLevel(logging.INFO)

    # If not already configured, add a file handler and an in-memory handler.
    if not conv_logger.handlers:
        # Create a directory for conversation logs if it doesn't exist.
        conv_log_dir = "./conversation_logs"
        if not os.path.exists(conv_log_dir):
            os.makedirs(conv_log_dir)
        conv_log_path = os.path.join(conv_log_dir, f"{conversation_id}.log")

        # File handler.
        fh = logging.FileHandler(conv_log_path, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        conv_logger.addHandler(fh)

        # In-memory log handler.
        mem_handler = InMemoryLogHandler()
        mem_handler.setLevel(logging.INFO)
        mem_handler.setFormatter(formatter)
        conv_logger.addHandler(mem_handler)
    else:
        # If configured, retrieve the existing in-memory handler (or create one if missing).
        mem_handler = None
        for handler in conv_logger.handlers:
            if isinstance(handler, InMemoryLogHandler):
                mem_handler = handler
                break
        if not mem_handler:
            mem_handler = InMemoryLogHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            mem_handler.setFormatter(formatter)
            conv_logger.addHandler(mem_handler)

    return conv_logger, mem_handler