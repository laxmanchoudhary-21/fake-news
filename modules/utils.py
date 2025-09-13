"""
Utility functions for the misinformation detection platform.
"""

import asyncio
import hashlib
import re
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import streamlit as st
from cachetools import TTLCache
import diskcache as dc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global caches
_memory_cache = TTLCache(maxsize=1000, ttl=3600)
_disk_cache = dc.Cache('.cache')

def setup_cache():
    """Initialize cache systems."""
    try:
        # Ensure cache directory exists
        import os
        os.makedirs('.cache', exist_ok=True)
        logger.info("Cache system initialized")
    except Exception as e:
        logger.error(f"Cache initialization failed: {e}")

def cache_result(key: str, ttl: int = 3600, use_disk: bool = False):
    """
    Decorator for caching function results.
    
    Args:
        key: Cache key template
        ttl: Time to live in seconds
        use_disk: Whether to use disk cache
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key}_{hash_args(args, kwargs)}"
            
            # Choose cache backend
            cache = _disk_cache if use_disk else _memory_cache
            
            # Try to get from cache
            try:
                if cache_key in cache:
                    logger.info(f"Cache hit for {cache_key}")
                    return cache[cache_key]
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            try:
                if use_disk:
                    cache.set(cache_key, result, expire=ttl)
                else:
                    cache[cache_key] = result
                logger.info(f"Cached result for {cache_key}")
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
            
            return result
        return wrapper
    return decorator

def hash_args(*args, **kwargs) -> str:
    """Generate a hash from function arguments."""
    content = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(content.encode()).hexdigest()[:8]

def clean_text(text: str) -> str:
    """
    Clean and normalize text for analysis.
    
    Args:
        text: Raw text input
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Normalize quotes and dashes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('–', '-').replace('—', '-')
    
    # Remove URLs (optional - preserve for analysis)
    # text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Clean up extra spaces again
    text = ' '.join(text.split())
    
    return text.strip()

def extract_keywords(text: str, max_words: int = 5) -> List[str]:
    """
    Extract important keywords from text.
    
    Args:
        text: Input text
        max_words: Maximum number of keywords
        
    Returns:
        List of keywords
    """
    if not text:
        return []
    
    # Common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 
        'may', 'might', 'must', 'this', 'that', 'these', 'those', 'i', 'you', 
        'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter stop words and get unique keywords
    keywords = []
    seen = set()
    for word in words:
        if word not in stop_words and word not in seen:
            keywords.append(word)
            seen.add(word)
        if len(keywords) >= max_words:
            break
    
    return keywords

def format_confidence(confidence: float) -> str:
    """Format confidence score for display."""
    return f"{confidence * 100:.1f}%"

def format_duration(seconds: float) -> str:
    """Format duration for display."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""
    try:
        return numerator / denominator if denominator != 0 else default
    except (ZeroDivisionError, TypeError):
        return default

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def validate_url(url: str) -> bool:
    """Validate if string is a valid URL."""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove or replace dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext
    return sanitized

def calculate_text_stats(text: str) -> Dict[str, Any]:
    """Calculate basic text statistics."""
    if not text:
        return {
            "word_count": 0,
            "char_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0,
            "avg_sentence_length": 0
        }
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return {
        "word_count": len(words),
        "char_count": len(text),
        "sentence_count": len(sentences),
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
        "avg_sentence_length": len(words) / len(sentences) if sentences else 0
    }

def get_file_type(filename: str) -> str:
    """Determine file type from extension."""
    from .config import SUPPORTED_FORMATS
    
    ext = os.path.splitext(filename)[1].lower()
    
    for file_type, extensions in SUPPORTED_FORMATS.items():
        if ext in extensions:
            return file_type
    
    return "unknown"

def format_file_size(size_bytes: int) -> str:
    """Format file size for display."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"

async def run_with_timeout(coro, timeout: float):
    """Run coroutine with timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        raise

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for retrying failed operations.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
                        
            raise last_exception
        return wrapper
    return decorator

def create_progress_bar(total: int, description: str = "Processing"):
    """Create a Streamlit progress bar."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update(current: int, message: str = ""):
        progress = min(current / total, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"{description}: {current}/{total} {message}")
    
    return update

def generate_report_id() -> str:
    """Generate unique report ID."""
    timestamp = str(int(time.time()))
    random_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    return f"report_{timestamp}_{random_hash}"

def is_suspicious_pattern(text: str) -> Dict[str, bool]:
    """
    Detect suspicious patterns in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of pattern detection results
    """
    if not text:
        return {}
    
    text_lower = text.lower()
    
    patterns = {
        "clickbait": any(phrase in text_lower for phrase in [
            "you won't believe", "shocking truth", "doctors hate this",
            "one weird trick", "this will blow your mind", "click here",
            "must see", "gone wrong", "what happens next"
        ]),
        "conspiracy": any(phrase in text_lower for phrase in [
            "they don't want you to know", "hidden truth", "cover up",
            "mainstream media", "wake up", "open your eyes",
            "government conspiracy", "big pharma", "illuminati"
        ]),
        "medical_misinformation": any(phrase in text_lower for phrase in [
            "cure cancer", "miracle cure", "natural remedy",
            "big pharma doesn't want", "doctors suppressed",
            "alternative medicine", "detox", "cleanse"
        ]),
        "excessive_caps": len([c for c in text if c.isupper()]) / len(text) > 0.3 if text else False,
        "excessive_punctuation": text.count('!') > 5 or text.count('?') > 3,
        "urgent_language": any(phrase in text_lower for phrase in [
            "urgent", "breaking", "alert", "warning",
            "immediately", "right now", "don't wait"
        ])
    }
    
    return patterns

def calculate_readability_score(text: str) -> float:
    """
    Calculate text readability score (simplified Flesch Reading Ease).
    
    Args:
        text: Text to analyze
        
    Returns:
        Readability score (0-100, higher is more readable)
    """
    if not text:
        return 0.0
    
    # Count sentences, words, and syllables
    sentences = len(re.split(r'[.!?]+', text))
    words = len(text.split())
    
    if sentences == 0 or words == 0:
        return 0.0
    
    # Estimate syllables (simplified)
    syllables = 0
    for word in text.split():
        word = word.lower().strip('.,!?";')
        syllables += max(1, len(re.findall(r'[aeiouAEIOU]', word)))
    
    # Flesch Reading Ease formula (simplified)
    score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
    
    return max(0, min(100, score))

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from text (simplified version).
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of entity types and their values
    """
    if not text:
        return {}
    
    entities = {
        "urls": re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text),
        "emails": re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
        "dates": re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', text),
        "numbers": re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text),
        "capitalized": re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    }
    
    return {k: list(set(v)) for k, v in entities.items() if v}

# Initialize cache on import
setup_cache()