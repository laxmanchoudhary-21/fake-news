"""
Configuration and constants for the misinformation detection platform.
"""

import os
from typing import Dict, List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys (optional - app works without them)
    newsapi_key: Optional[str] = os.getenv("NEWSAPI_KEY", "")
    newsdata_key: Optional[str] = os.getenv("NEWSDATA_KEY", "")
    google_factcheck_key: Optional[str] = os.getenv("GOOGLE_FACTCHECK_KEY", "")
    
    # Model configurations
    text_model_name: str = "martin-ha/toxic-comment-model"
    sentence_model_name: str = "all-MiniLM-L6-v2"
    face_detector_model: str = "yunet"
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    cache_maxsize: int = 1000
    
    # API timeouts and limits
    api_timeout: int = 30
    max_concurrent_requests: int = 10
    rate_limit_calls: int = 100
    rate_limit_period: int = 3600
    
    # Processing limits
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    max_text_length: int = 10000
    max_video_duration: int = 300  # 5 minutes
    max_video_frames: int = 30
    
    # UI settings
    default_theme: str = "dark"
    enable_animations: bool = True
    show_debug_info: bool = False
    
    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()

# Model configurations
MODEL_CONFIGS = {
    "text_classification": {
        "models": [
            "martin-ha/toxic-comment-model",
            "unitary/toxic-bert",
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        ],
        "fallback_threshold": 0.5
    },
    "sentence_embedding": {
        "model": "all-MiniLM-L6-v2",
        "similarity_threshold": 0.8
    },
    "face_detection": {
        "yunet_url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "cascade_path": "haarcascade_frontalface_default.xml"
    }
}

# Data source configurations
DATA_SOURCES = {
    "rss_feeds": [
        ("BBC News", "http://feeds.bbci.co.uk/news/rss.xml"),
        ("Reuters", "http://feeds.reuters.com/reuters/topNews"),
        ("AP News", "https://feeds.apnews.com/rss/apf-topnews"),
        ("NPR", "https://feeds.npr.org/1001/rss.xml"),
        ("CNN", "http://rss.cnn.com/rss/edition.rss"),
        ("The Guardian", "https://www.theguardian.com/world/rss"),
        ("Associated Press", "https://feeds.apnews.com/rss/apf-topnews"),
        ("ABC News", "https://abcnews.go.com/abcnews/topstories")
    ],
    "fact_checkers": [
        "snopes.com",
        "politifact.com", 
        "factcheck.org",
        "checkyourfact.com",
        "truthorfiction.com"
    ]
}

# Supported file formats
SUPPORTED_FORMATS = {
    "images": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
    "videos": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"],
    "documents": [".pdf", ".txt", ".docx", ".doc"],
    "archives": [".zip", ".tar", ".gz"]
}

# UI Theme configurations
THEMES = {
    "dark": {
        "primary_bg": "#0E1117",
        "secondary_bg": "#1E2329", 
        "card_bg": "#262730",
        "accent_bg": "#2D3138",
        "text_primary": "#FFFFFF",
        "text_secondary": "#A9A9A9",
        "accent_color": "#00D4AA",
        "gradient_primary": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "gradient_secondary": "linear-gradient(135deg, #00D4AA 0%, #00A085 100%)",
        "gradient_danger": "linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%)",
        "gradient_warning": "linear-gradient(135deg, #feca57 0%, #ff9ff3 100%)",
        "gradient_success": "linear-gradient(135deg, #48CAE4 0%, #00D4AA 100%)",
        "border_color": "#3D4147",
        "shadow": "0 8px 32px rgba(0, 0, 0, 0.3)",
        "glow": "0 0 20px rgba(0, 212, 170, 0.3)"
    },
    "light": {
        "primary_bg": "#FFFFFF",
        "secondary_bg": "#F8F9FA",
        "card_bg": "#FFFFFF",
        "accent_bg": "#F1F3F4",
        "text_primary": "#1A1A1A",
        "text_secondary": "#6C757D",
        "accent_color": "#007BFF",
        "gradient_primary": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "gradient_secondary": "linear-gradient(135deg, #007BFF 0%, #0056B3 100%)",
        "gradient_danger": "linear-gradient(135deg, #DC3545 0%, #B02A37 100%)",
        "gradient_warning": "linear-gradient(135deg, #FFC107 0%, #E0A800 100%)",
        "gradient_success": "linear-gradient(135deg, #28A745 0%, #1E7E34 100%)",
        "border_color": "#E9ECEF",
        "shadow": "0 8px 32px rgba(0, 0, 0, 0.1)",
        "glow": "0 0 20px rgba(0, 123, 255, 0.2)"
    }
}

# Analysis thresholds
ANALYSIS_THRESHOLDS = {
    "fake_news": {
        "high_confidence": 0.8,
        "medium_confidence": 0.6,
        "low_confidence": 0.4
    },
    "deepfake": {
        "high_risk": 0.7,
        "medium_risk": 0.4,
        "low_risk": 0.2
    },
    "verification": {
        "high_verification": 0.7,
        "medium_verification": 0.4,
        "low_verification": 0.2
    }
}

# Example data for demos
DEMO_EXAMPLES = {
    "fake_news": [
        "Scientists discover that drinking coffee can make you immortal, study shows",
        "Breaking: Aliens confirmed to have built the pyramids, NASA reveals",
        "New research proves that vaccines contain microchips for mind control"
    ],
    "real_news": [
        "Stock markets close mixed as investors await Federal Reserve decision",
        "Research shows moderate coffee consumption may have health benefits",
        "Climate change continues to affect global weather patterns, scientists report"
    ],
    "suspicious_claims": [
        "This one weird trick will cure all diseases doctors don't want you to know",
        "Local mom discovers anti-aging secret that dermatologists hate",
        "Government tries to hide this shocking truth about water fluoridation"
    ]
}

# Error messages
ERROR_MESSAGES = {
    "model_load_failed": "Failed to load AI model. Using fallback detection.",
    "api_timeout": "External service timeout. Using cached results.",
    "file_too_large": f"File size exceeds {settings.max_file_size // (1024*1024)}MB limit.",
    "invalid_format": "Unsupported file format. Please use supported formats.",
    "network_error": "Network connection issue. Some features may be limited.",
    "quota_exceeded": "API quota exceeded. Using alternative sources."
}

# Success messages  
SUCCESS_MESSAGES = {
    "analysis_complete": "Analysis completed successfully!",
    "file_uploaded": "File uploaded and processed successfully.",
    "cache_hit": "Results loaded from cache for faster response.",
    "fallback_success": "Using alternative verification sources.",
    "model_loaded": "AI models loaded successfully."
}