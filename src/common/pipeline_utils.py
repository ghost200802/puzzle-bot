"""
Pipeline utilities: progress reporting, logging, error recovery.
"""

import os
import sys
import time
import logging
import functools
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('puzzle-bot')


class ProgressBar:
    """
    Simple text-based progress bar.
    """

    def __init__(self, total, prefix='', suffix='', bar_length=40):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.bar_length = bar_length
        self.current = 0
        self.start_time = time.time()

    def update(self, current=None):
        if current is not None:
            self.current = current
        else:
            self.current += 1

        if self.total == 0:
            return

        fraction = self.current / self.total
        arrow_count = int(fraction * self.bar_length)
        arrow = '#' * arrow_count
        spaces = '-' * (self.bar_length - arrow_count)
        elapsed = time.time() - self.start_time
        if self.current > 0 and self.current < self.total:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = f' ETA: {eta:.0f}s'
        elif self.current >= self.total:
            eta_str = f' Done: {elapsed:.1f}s'
        else:
            eta_str = ''

        pct = fraction * 100
        sys.stdout.write(
            f'\r{self.prefix} |{arrow}{spaces}| {pct:5.1f}% {self.suffix}{eta_str}'
        )
        sys.stdout.flush()

        if self.current >= self.total:
            sys.stdout.write('\n')
            sys.stdout.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.current < self.total:
            sys.stdout.write('\n')


def retry(max_retries=3, delay=1.0, backoff=2.0, exceptions=(Exception,)):
    """
    Decorator for retrying a function on failure.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise
                    logger.warning(
                        f"{func.__name__} failed (attempt {retries}/{max_retries}): {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator


def safe_execute(func, *args, default=None, log_errors=True, **kwargs):
    """
    Execute a function safely, returning default on failure.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Error in {func.__name__}: {e}")
            logger.debug(traceback.format_exc())
        return default


class PipelineCheckpoint:
    """
    Simple file-based checkpoint for pipeline progress.
    Allows resuming from the last completed step.
    """

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, '.checkpoint')
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, step, data=None):
        """Save current progress."""
        import json
        state = {'step': step, 'data': data}
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f)
        logger.info(f"Checkpoint saved: step={step}")

    def load(self):
        """Load last checkpoint. Returns (step, data) or (0, None)."""
        import json
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file) as f:
                state = json.load(f)
            return state.get('step', 0), state.get('data')
        return 0, None

    def clear(self):
        """Remove checkpoint file."""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)

    def is_complete(self, step):
        """Check if a step has been completed."""
        saved_step, _ = self.load()
        return saved_step >= step


def log_step(step_num, step_name):
    """
    Decorator that logs pipeline step execution.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Step {step_num}: {step_name} - Starting...")
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                logger.info(
                    f"Step {step_num}: {step_name} - Completed in {duration:.1f}s"
                )
                return result
            except Exception as e:
                duration = time.time() - start
                logger.error(
                    f"Step {step_num}: {step_name} - Failed after {duration:.1f}s: {e}"
                )
                raise
        return wrapper
    return decorator
