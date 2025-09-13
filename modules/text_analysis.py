"""
Advanced text analysis module with improved models and explainability.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import streamlit as st

# Core ML libraries
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

# Explainability
try:
    import shap
    import lime
    from lime.lime_text import LimeTextExplainer
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False
    st.warning("⚠️ SHAP/LIME not available. Install with: pip install shap lime")

# NLP utilities
try:
    import nltk
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    from wordcloud import WordCloud
    NLP_ENHANCED = True
except ImportError:
    NLP_ENHANCED = False

from .config import MODEL_CONFIGS, ANALYSIS_THRESHOLDS, settings
from .utils import cache_result, clean_text, extract_keywords, logger

class TextAnalyzer:
    """Advanced text analysis with multiple models and explainability."""
    
    def __init__(self):
        self.classification_models = {}
        self.sentence_model = None
        self.tokenizer = None
        self.lime_explainer = None
        self.shap_explainer = None
        self._initialize_models()
    
    @cache_result("model_init", ttl=86400)  # Cache for 24 hours
    def _initialize_models(self):
        """Initialize all text analysis models."""
        with st.spinner("🤖 Loading AI models..."):
            self._load_classification_models()
            self._load_sentence_model()
            self._setup_explainability()
    
    def _load_classification_models(self):
        """Load text classification models with fallbacks."""
        models_config = MODEL_CONFIGS["text_classification"]["models"]
        
        for i, model_name in enumerate(models_config):
            try:
                st.info(f"Loading model {i+1}/{len(models_config)}: {model_name}")
                
                # Load model with error handling
                model = pipeline(
                    "text-classification",
                    model=model_name,
                    return_all_scores=True,
                    truncation=True,
                    max_length=512,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Test model with sample input
                test_result = model("This is a test message.")
                if test_result:
                    self.classification_models[model_name] = model
                    st.success(f"✅ Loaded: {model_name}")
                    
                    # Use first working model as primary
                    if 'primary' not in self.classification_models:
                        self.classification_models['primary'] = model
                        # Also load tokenizer for explanations
                        try:
                            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        except Exception as e:
                            logger.warning(f"Could not load tokenizer: {e}")
                    break
                    
            except Exception as e:
                st.warning(f"⚠️ Failed to load {model_name}: {str(e)[:100]}...")
                continue
        
        if not self.classification_models:
            st.error("❌ No text classification models could be loaded")
    
    def _load_sentence_model(self):
        """Load sentence transformer for semantic similarity."""
        try:
            model_name = MODEL_CONFIGS["sentence_embedding"]["model"]
            st.info(f"Loading sentence transformer: {model_name}")
            
            self.sentence_model = SentenceTransformer(model_name)
            st.success(f"✅ Sentence transformer loaded")
            
        except Exception as e:
            st.warning(f"⚠️ Sentence transformer loading failed: {e}")
            self.sentence_model = None
    
    def _setup_explainability(self):
        """Setup SHAP and LIME explainers."""
        if not EXPLAINABILITY_AVAILABLE:
            return
        
        try:
            # Setup LIME explainer
            self.lime_explainer = LimeTextExplainer(
                class_names=['authentic', 'fake'],
                feature_selection='auto',
                split_expression=r'\W+',
                bow=True
            )
            
            # SHAP explainer will be initialized per prediction
            st.info("✅ Explainability tools ready")
            
        except Exception as e:
            logger.warning(f"Could not setup explainability: {e}")
    
    def analyze_text(self, text: str, explain: bool = True) -> Dict[str, Any]:
        """
        Comprehensive text analysis with explainability.
        
        Args:
            text: Input text to analyze
            explain: Whether to generate explanations
            
        Returns:
            Analysis results with predictions and explanations
        """
        if not text:
            return {"error": "No text provided"}
        
        # Clean and prepare text
        cleaned_text = clean_text(text)
        if len(cleaned_text) > settings.max_text_length:
            cleaned_text = cleaned_text[:settings.max_text_length]
        
        results = {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "text_stats": self._calculate_text_statistics(cleaned_text),
            "suspicious_patterns": self._detect_suspicious_patterns(cleaned_text),
            "classification": {},
            "explanations": {},
            "similarity_analysis": {}
        }
        
        # Run classification
        if 'primary' in self.classification_models:
            results["classification"] = self._classify_text(cleaned_text)
            
            # Generate explanations if requested
            if explain and EXPLAINABILITY_AVAILABLE:
                results["explanations"] = self._generate_explanations(cleaned_text)
        
        # Semantic similarity analysis
        if self.sentence_model:
            results["similarity_analysis"] = self._analyze_similarity(cleaned_text)
        
        return results
    
    @cache_result("text_classification", ttl=3600)
    def _classify_text(self, text: str) -> Dict[str, Any]:
        """Classify text as fake or authentic news."""
        try:
            model = self.classification_models['primary']
            result = model(text)
            
            # Process results based on model output format
            if isinstance(result, list) and len(result) > 0:
                predictions = result[0] if isinstance(result[0], list) else result
                
                # Extract fake/real probabilities
                fake_score = 0.0
                real_score = 0.0
                
                for pred in predictions:
                    label = pred['label'].lower()
                    score = pred['score']
                    
                    # Map various label formats to fake/real
                    if any(indicator in label for indicator in ['fake', 'false', 'toxic', 'negative', '1']):
                        fake_score = max(fake_score, score)
                    elif any(indicator in label for indicator in ['real', 'true', 'non-toxic', 'positive', '0']):
                        real_score = max(real_score, score)
                
                # Determine final prediction
                if fake_score > real_score:
                    prediction = "fake"
                    confidence = fake_score
                else:
                    prediction = "authentic"
                    confidence = real_score
                
                # Determine confidence level
                if confidence >= ANALYSIS_THRESHOLDS["fake_news"]["high_confidence"]:
                    confidence_level = "high"
                elif confidence >= ANALYSIS_THRESHOLDS["fake_news"]["medium_confidence"]:
                    confidence_level = "medium"
                else:
                    confidence_level = "low"
                
                return {
                    "prediction": prediction,
                    "confidence": float(confidence),
                    "confidence_level": confidence_level,
                    "fake_probability": float(fake_score),
                    "authentic_probability": float(real_score),
                    "raw_predictions": predictions,
                    "model_used": list(self.classification_models.keys())[0]
                }
            
        except Exception as e:
            logger.error(f"Text classification failed: {e}")
            return {"error": str(e)}
    
    def _generate_explanations(self, text: str) -> Dict[str, Any]:
        """Generate SHAP and LIME explanations for predictions."""
        explanations = {}
        
        try:
            # LIME explanation
            if self.lime_explainer and 'primary' in self.classification_models:
                
                def predict_proba(texts):
                    """Prediction function for LIME."""
                    results = []
                    for txt in texts:
                        try:
                            pred = self.classification_models['primary'](txt)
                            if isinstance(pred, list) and len(pred) > 0:
                                # Extract probabilities
                                fake_prob = 0.0
                                real_prob = 0.0
                                
                                preds = pred[0] if isinstance(pred[0], list) else pred
                                for p in preds:
                                    label = p['label'].lower()
                                    if any(ind in label for ind in ['fake', 'toxic', 'negative']):
                                        fake_prob = p['score']
                                    elif any(ind in label for ind in ['real', 'positive']):
                                        real_prob = p['score']
                                
                                results.append([real_prob, fake_prob])
                            else:
                                results.append([0.5, 0.5])
                        except Exception:
                            results.append([0.5, 0.5])
                    return np.array(results)
                
                # Generate LIME explanation
                lime_exp = self.lime_explainer.explain_instance(
                    text, 
                    predict_proba, 
                    num_features=10,
                    num_samples=1000
                )
                
                # Extract feature importance
                lime_features = lime_exp.as_list()
                explanations["lime"] = {
                    "features": lime_features,
                    "html": lime_exp.as_html() if hasattr(lime_exp, 'as_html') else None
                }
        
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
        
        # Add word importance based on attention (simplified)
        try:
            explanations["word_importance"] = self._calculate_word_importance(text)
        except Exception as e:
            logger.warning(f"Word importance calculation failed: {e}")
        
        return explanations
    
    def _calculate_word_importance(self, text: str) -> List[Tuple[str, float]]:
        """Calculate word importance scores (simplified attention mechanism)."""
        words = text.split()
        
        # Simplified importance based on common fake news patterns
        suspicious_words = {
            'shocking', 'unbelievable', 'secret', 'hidden', 'truth', 'exposed',
            'exclusive', 'breaking', 'urgent', 'alert', 'warning', 'conspiracy',
            'cover-up', 'leaked', 'revealed', 'insider', 'anonymous', 'sources'
        }
        
        important_words = []
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            # Base importance
            importance = 0.1
            
            # Higher importance for suspicious words
            if clean_word in suspicious_words:
                importance = 0.8
            
            # Higher importance for capitalized words
            if word.isupper() and len(word) > 2:
                importance += 0.3
            
            # Higher importance for longer words
            if len(clean_word) > 8:
                importance += 0.2
            
            important_words.append((word, min(importance, 1.0)))
        
        return important_words
    
    def _analyze_similarity(self, text: str) -> Dict[str, Any]:
        """Analyze semantic similarity with known patterns."""
        if not self.sentence_model:
            return {}
        
        try:
            # Encode the input text
            text_embedding = self.sentence_model.encode([text])
            
            # Compare with known fake news patterns (this would be enhanced with a real database)
            fake_patterns = [
                "Scientists discovered a miracle cure that doctors don't want you to know about",
                "Breaking news: Government hiding shocking truth from citizens",
                "This one simple trick will solve all your problems instantly"
            ]
            
            real_patterns = [
                "Research published in peer-reviewed journal shows promising results",
                "According to official government statistics released today",
                "Study conducted by university researchers indicates correlation"
            ]
            
            # Calculate similarities
            fake_embeddings = self.sentence_model.encode(fake_patterns)
            real_embeddings = self.sentence_model.encode(real_patterns)
            
            fake_similarities = F.cosine_similarity(
                torch.tensor(text_embedding), 
                torch.tensor(fake_embeddings)
            )
            real_similarities = F.cosine_similarity(
                torch.tensor(text_embedding), 
                torch.tensor(real_embeddings)
            )
            
            return {
                "fake_pattern_similarity": float(torch.max(fake_similarities)),
                "real_pattern_similarity": float(torch.max(real_similarities)),
                "most_similar_fake": fake_patterns[torch.argmax(fake_similarities)],
                "most_similar_real": real_patterns[torch.argmax(real_similarities)]
            }
            
        except Exception as e:
            logger.error(f"Similarity analysis failed: {e}")
            return {}
    
    def _calculate_text_statistics(self, text: str) -> Dict[str, Any]:
        """Calculate comprehensive text statistics."""
        if not text:
            return {}
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        stats = {
            "word_count": len(words),
            "character_count": len(text),
            "sentence_count": len(sentences),
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "unique_words": len(set(word.lower() for word in words)),
            "lexical_diversity": len(set(word.lower() for word in words)) / len(words) if words else 0
        }
        
        # Enhanced statistics if textstat is available
        if NLP_ENHANCED:
            try:
                stats.update({
                    "readability_score": flesch_reading_ease(text),
                    "grade_level": flesch_kincaid_grade(text),
                })
            except Exception as e:
                logger.warning(f"Enhanced text statistics failed: {e}")
        
        return stats
    
    def _detect_suspicious_patterns(self, text: str) -> Dict[str, bool]:
        """Detect various suspicious patterns in text."""
        if not text:
            return {}
        
        text_lower = text.lower()
        
        patterns = {
            "clickbait_language": any(phrase in text_lower for phrase in [
                "you won't believe", "shocking truth", "doctors hate this",
                "one weird trick", "this will blow your mind", "click here",
                "must see", "gone wrong", "what happens next", "number 7 will shock you"
            ]),
            
            "conspiracy_language": any(phrase in text_lower for phrase in [
                "they don't want you to know", "hidden truth", "cover up",
                "mainstream media lies", "wake up sheeple", "open your eyes",
                "government conspiracy", "big pharma", "illuminati", "deep state"
            ]),
            
            "medical_misinformation": any(phrase in text_lower for phrase in [
                "cure cancer", "miracle cure", "natural remedy cures everything",
                "big pharma doesn't want", "doctors suppressed this",
                "alternative medicine breakthrough", "toxins", "cleanse"
            ]),
            
            "fake_urgency": any(phrase in text_lower for phrase in [
                "urgent", "breaking news", "alert", "immediate action required",
                "time is running out", "act now", "don't wait", "limited time"
            ]),
            
            "emotional_manipulation": any(phrase in text_lower for phrase in [
                "you'll be furious", "this makes me sick", "absolutely disgusting",
                "heartbreaking", "outrageous", "unforgivable", "criminal"
            ]),
            
            "false_authority": any(phrase in text_lower for phrase in [
                "scientists say", "experts agree", "studies show", "research proves",
                "according to sources", "insiders reveal", "leaked documents"
            ]),
            
            "excessive_capitalization": len([c for c in text if c.isupper()]) / len(text) > 0.3 if text else False,
            "excessive_punctuation": text.count('!') > 5 or text.count('?') > 3,
            "suspicious_urls": len(re.findall(r'bit\.ly|tinyurl|short\.link', text_lower)) > 0,
            "all_caps_words": len([word for word in text.split() if word.isupper() and len(word) > 2]) > 3
        }
        
        return patterns
    
    def generate_wordcloud(self, text: str) -> Optional[Any]:
        """Generate word cloud for text visualization."""
        if not NLP_ENHANCED or not text:
            return None
        
        try:
            # Clean text for word cloud
            clean_words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            clean_text = ' '.join(clean_words)
            
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(clean_text)
            
            return wordcloud
            
        except Exception as e:
            logger.warning(f"Word cloud generation failed: {e}")
            return None

# Global text analyzer instance
_text_analyzer = None

def get_text_analyzer() -> TextAnalyzer:
    """Get or create global text analyzer instance."""
    global _text_analyzer
    if _text_analyzer is None:
        _text_analyzer = TextAnalyzer()
    return _text_analyzer