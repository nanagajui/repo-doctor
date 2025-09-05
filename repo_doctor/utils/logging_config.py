"""Structured logging configuration for Repo Doctor - STREAM B Enhancement."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import traceback


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_obj.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_obj)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m',     # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Get color for level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Build colored message
        if record.levelname in ['ERROR', 'CRITICAL']:
            # Show more detail for errors
            return (f"{color}[{timestamp}] {record.levelname}{reset} "
                   f"[{record.name}:{record.funcName}:{record.lineno}] "
                   f"{record.getMessage()}")
        else:
            # Simpler format for info/debug
            return (f"{color}[{timestamp}] {record.levelname}{reset} "
                   f"[{record.name}] {record.getMessage()}")


class RepoDocLog:
    """Centralized logging configuration for Repo Doctor."""
    
    _initialized = False
    _logger_cache = {}
    
    @classmethod
    def setup(
        cls,
        level: str = "INFO",
        log_file: Optional[str] = None,
        json_format: bool = False,
        console: bool = True,
    ) -> None:
        """
        Set up logging configuration.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
            json_format: Use JSON structured logging format
            console: Enable console output
        """
        if cls._initialized:
            return
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        root_logger.handlers = []
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            if json_format:
                console_handler.setFormatter(StructuredFormatter())
            else:
                console_handler.setFormatter(ColoredFormatter())
            console_handler.setLevel(getattr(logging, level.upper()))
            root_logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            # Always use JSON format for files
            file_handler.setFormatter(StructuredFormatter())
            file_handler.setLevel(getattr(logging, level.upper()))
            root_logger.addHandler(file_handler)
        
        # Set third-party library log levels to reduce noise
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("github").setLevel(logging.WARNING)
        logging.getLogger("docker").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger instance with the given name.
        
        Args:
            name: Logger name (usually __name__)
            
        Returns:
            Logger instance
        """
        if not cls._initialized:
            cls.setup()
        
        if name not in cls._logger_cache:
            cls._logger_cache[name] = logging.getLogger(name)
        
        return cls._logger_cache[name]
    
    @classmethod
    def log_with_context(
        cls,
        logger: logging.Logger,
        level: str,
        message: str,
        **context
    ) -> None:
        """
        Log a message with additional context fields.
        
        Args:
            logger: Logger instance
            level: Log level
            message: Log message
            **context: Additional context fields
        """
        # Create a LogRecord with extra fields
        record = logger.makeRecord(
            logger.name,
            getattr(logging, level.upper()),
            "(unknown file)",
            0,
            message,
            (),
            None,
        )
        record.extra_fields = context
        logger.handle(record)


# Convenience functions
def get_logger(name: str = __name__) -> logging.Logger:
    """Get a logger instance."""
    return RepoDocLog.get_logger(name)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    console: bool = True,
) -> None:
    """Set up logging configuration."""
    RepoDocLog.setup(level, log_file, json_format, console)


def log_function_call(func_name: str, **kwargs):
    """Log a function call with parameters."""
    logger = get_logger("repo_doctor.trace")
    logger.debug(f"Calling {func_name}", extra={"function": func_name, "params": kwargs})


def log_performance(operation: str, duration: float, **metadata):
    """Log performance metrics."""
    logger = get_logger("repo_doctor.performance")
    logger.info(
        f"{operation} completed in {duration:.2f}s",
        extra={"operation": operation, "duration": duration, **metadata}
    )


def log_error(error: Exception, context: str = "", **metadata):
    """Log an error with context."""
    logger = get_logger("repo_doctor.error")
    logger.error(
        f"Error in {context}: {str(error)}",
        exc_info=error,
        extra={"context": context, **metadata}
    )