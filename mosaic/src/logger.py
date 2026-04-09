import logging
import os
from datetime import datetime, timezone


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


def log_pipeline_event(message: str) -> None:
    """
    流水线事件（仅时间戳 + 一行说明，不含 prompt）。
    设置 MOSAIC_VERBOSE_LOG 为文件路径即写入（与子进程是否 --verbose 无关）。
    """
    path = os.environ.get("MOSAIC_VERBOSE_LOG", "").strip()
    if not path:
        return
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts}\t{message}\n"
    try:
        d = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(d, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def setup_logger(name: str) -> logging.Logger:
    """
    设置 logger。
    控制台：
      - 未设 MOSAIC_VERBOSE=1 且未显式覆盖时：WARNING（避免刷屏；构图进度请用 tqdm 或进度文件）
      - MOSAIC_VERBOSE=1：DEBUG（与 --verbose 一致）
      - MOSAIC_CONSOLE_MIN_LEVEL 优先于上述规则（兼容服务器脚本）
    文件：
      - 默认 INFO（不写 DEBUG 级 prompt/TF-IDF 大块文本）；MOSAIC_LOG_FILE_LEVEL=DEBUG 恢复全文调试日志
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler()
        min_console = os.environ.get("MOSAIC_CONSOLE_MIN_LEVEL", "").strip().upper()
        if min_console and hasattr(logging, min_console):
            ch.setLevel(getattr(logging, min_console))
        elif os.environ.get("MOSAIC_VERBOSE") == "1":
            ch.setLevel(logging.DEBUG)
        else:
            ch.setLevel(logging.WARNING)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # 文件：记录 DEBUG；目录由 MOSAIC_LOG_DIR（绝对路径）覆盖，否则 mosaic/log
        _mosaic_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.environ.get("MOSAIC_LOG_DIR", "").strip()
        if log_dir:
            log_dir = os.path.abspath(log_dir)
        else:
            log_dir = os.path.join(_mosaic_root, "log")
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_name = os.environ.get("MOSAIC_SERVER_LOG_BASENAME", "server.log").strip() or "server.log"
            log_path = os.path.join(log_dir, log_name)
        except OSError:
            log_dir = os.path.join(os.getcwd(), "log")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "server.log")

        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_lvl_name = os.environ.get("MOSAIC_LOG_FILE_LEVEL", "INFO").strip().upper()
        file_lvl = getattr(logging, file_lvl_name, logging.INFO)
        fh.setLevel(file_lvl)
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
        conv_log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conversation_logs")
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