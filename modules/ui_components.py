"""
Premium UI components with dark/light themes and interactive dashboards.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple 
import base64

from .config import THEMES, settings

class ThemeManager:
    """Manage application themes and styling."""
    
    def __init__(self):
        self.current_theme = self._get_current_theme()
    
    def _get_current_theme(self) -> str:
        """Get current theme from session state."""
        if 'theme' not in st.session_state:
            st.session_state.theme = settings.default_theme
        return st.session_state.theme
    
    def toggle_theme(self):
        """Toggle between light and dark themes."""
        current = st.session_state.get('theme', 'dark')
        st.session_state.theme = 'light' if current == 'dark' else 'dark'
        self.current_theme = st.session_state.theme
    
    def get_theme_colors(self) -> Dict[str, str]:
        """Get color palette for current theme."""
        return THEMES[self.current_theme]
    
    def apply_theme_css(self):
        """Apply theme-specific CSS to the app."""
        colors = self.get_theme_colors()
        
        css = f"""
        <style>
            /* Import Google Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
            
            /* Global Styles */
            .stApp {{
                background: {colors['primary_bg']};
                color: {colors['text_primary']};
                font-family: 'Inter', sans-serif;
            }}
            
            /* Hide Streamlit elements */
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            .stDeployButton {{display: none;}}
            header {{visibility: hidden;}}
            
            /* Custom Scrollbar */
            ::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}
            ::-webkit-scrollbar-track {{
                background: {colors['secondary_bg']};
                border-radius: 10px;
            }}
            ::-webkit-scrollbar-thumb {{
                background: {colors['gradient_secondary']};
                border-radius: 10px;
            }}
            ::-webkit-scrollbar-thumb:hover {{
                background: {colors['accent_color']};
            }}
            
            /* Main Header */
            .main-header {{
                font-size: 3.5rem;
                font-weight: 800;
                text-align: center;
                background: {colors['gradient_primary']};
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin: 2rem 0;
                letter-spacing: -0.02em;
                animation: fadeIn 1s ease-out;
            }}
            
            .main-subtitle {{
                font-size: 1.2rem;
                text-align: center;
                color: {colors['text_secondary']};
                margin-bottom: 3rem;
                font-weight: 400;
            }}
            
            /* Section Headers */
            .section-header {{
                font-size: 2rem;
                font-weight: 700;
                color: {colors['text_primary']};
                margin: 3rem 0 2rem 0;
                padding: 0 0 1rem 0;
                border-bottom: 3px solid;
                border-image: {colors['gradient_secondary']} 1;
                position: relative;
            }}
            
            .section-header::before {{
                content: '';
                position: absolute;
                left: 0;
                bottom: -3px;
                width: 60px;
                height: 3px;
                background: {colors['gradient_primary']};
                border-radius: 2px;
            }}
            
            /* Premium Cards */
            .premium-card {{
                background: {colors['card_bg']};
                border-radius: 20px;
                padding: 2rem;
                box-shadow: {colors['shadow']};
                border: 1px solid {colors['border_color']};
                margin: 1.5rem 0;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }}
            
            .premium-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: {colors['gradient_primary']};
            }}
            
            .premium-card:hover {{
                transform: translateY(-5px);
                box-shadow: {colors['glow']};
            }}
            
            /* Result Boxes */
            .result-box {{
                padding: 2rem;
                border-radius: 16px;
                margin: 1.5rem 0;
                box-shadow: {colors['shadow']};
                position: relative;
                overflow: hidden;
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            }}
            
            .result-box:hover {{
                transform: scale(1.02);
            }}
            
            .fake-result {{
                background: {colors['gradient_danger']};
                color: white;
                border: none;
            }}
            
            .real-result {{
                background: {colors['gradient_success']};
                color: white;
                border: none;
            }}
            
            .warning-result {{
                background: {colors['gradient_warning']};
                color: white;
                border: none;
            }}
            
            /* Verification Score */
            .verification-score {{
                font-size: 1.4rem;
                font-weight: 700;
                text-align: center;
                padding: 2rem;
                border-radius: 16px;
                margin: 2rem 0;
                position: relative;
                overflow: hidden;
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            }}
            
            .verification-score:hover {{
                transform: scale(1.02);
            }}
            
            .high-confidence {{
                background: {colors['gradient_success']};
                color: white;
                box-shadow: 0 0 30px rgba(72, 202, 228, 0.4);
            }}
            
            .medium-confidence {{
                background: {colors['gradient_warning']};
                color: white;
                box-shadow: 0 0 30px rgba(254, 202, 87, 0.4);
            }}
            
            .low-confidence {{
                background: {colors['gradient_danger']};
                color: white;
                box-shadow: 0 0 30px rgba(255, 107, 107, 0.4);
            }}
            
            /* Info Box */
            .info-box {{
                background: {colors['card_bg']};
                border-radius: 16px;
                padding: 2rem;
                margin: 2rem 0;
                border-left: 5px solid;
                border-image: {colors['gradient_secondary']} 1;
                box-shadow: {colors['shadow']};
                position: relative;
            }}
            
            .info-box h3 {{
                color: {colors['text_primary']};
                margin-bottom: 1rem;
                font-weight: 600;
            }}
            
            .info-box strong {{
                color: {colors['accent_color']};
            }}
            
            /* Buttons */
            .stButton > button {{
                background: {colors['gradient_secondary']};
                color: white;
                border: none;
                border-radius: 12px;
                padding: 0.75rem 2rem;
                font-weight: 600;
                font-size: 1rem;
                transition: all 0.3s ease;
                box-shadow: {colors['shadow']};
            }}
            
            .stButton > button:hover {{
                transform: translateY(-2px);
                box-shadow: {colors['glow']};
            }}
            
            /* File Uploader */
            .stFileUploader > div {{
                background: {colors['card_bg']};
                border: 2px dashed {colors['accent_color']};
                border-radius: 16px;
                padding: 2rem;
                text-align: center;
                transition: all 0.3s ease;
            }}
            
            .stFileUploader > div:hover {{
                border-color: {colors['accent_color']};
                background: {colors['accent_bg']};
                transform: scale(1.02);
            }}
            
            /* Sidebar */
            .css-1d391kg {{
                background: {colors['secondary_bg']};
                border-right: 1px solid {colors['border_color']};
            }}
            
            /* Metrics */
            .stMetric {{
                background: {colors['card_bg']};
                padding: 1rem;
                border-radius: 12px;
                border: 1px solid {colors['border_color']};
                box-shadow: {colors['shadow']};
                transition: all 0.3s ease;
            }}
            
            .stMetric:hover {{
                transform: translateY(-2px);
                box-shadow: {colors['glow']};
            }}
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
                background: {colors['secondary_bg']};
                border-radius: 12px;
                padding: 0.5rem;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                background: transparent;
                border-radius: 8px;
                color: {colors['text_secondary']};
                font-weight: 500;
                transition: all 0.3s ease;
            }}
            
            .stTabs [aria-selected="true"] {{
                background: {colors['gradient_secondary']};
                color: white;
            }}
            
            /* Text Input */
            .stTextArea textarea {{
                background: {colors['card_bg']};
                color: {colors['text_primary']};
                border: 2px solid {colors['border_color']};
                border-radius: 12px;
                font-size: 1rem;
                transition: all 0.3s ease;
            }}
            
            .stTextArea textarea:focus {{
                border-color: {colors['accent_color']};
                box-shadow: {colors['glow']};
            }}
            
            /* Source Tags */
            .source-tag {{
                background: {colors['gradient_secondary']};
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 25px;
                font-size: 0.9rem;
                font-weight: 500;
                margin: 0.25rem;
                display: inline-block;
                box-shadow: {colors['shadow']};
                transition: all 0.3s ease;
            }}
            
            .source-tag:hover {{
                transform: translateY(-1px);
                box-shadow: {colors['glow']};
            }}
            
            /* Status Indicators */
            .status-success {{
                color: #00D4AA;
                font-weight: 600;
            }}
            
            .status-error {{
                color: #ff6b6b;
                font-weight: 600;
            }}
            
            .status-warning {{
                color: #feca57;
                font-weight: 600;
            }}
            
            /* Animations */
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            @keyframes pulse {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.05); }}
                100% {{ transform: scale(1); }}
            }}
            
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            .fade-in {{
                animation: fadeIn 0.5s ease-out;
            }}
            
            .pulse {{
                animation: pulse 2s infinite;
            }}
            
            .loading-spinner {{
                animation: spin 1s linear infinite;
            }}
            
            /* Mobile Responsive */
            @media (max-width: 768px) {{
                .main-header {{
                    font-size: 2.5rem;
                }}
                
                .section-header {{
                    font-size: 1.5rem;
                }}
                
                .premium-card {{
                    padding: 1.5rem;
                }}
                
                .result-box {{
                    padding: 1.5rem;
                }}
            }}
        </style>
        """
        
        st.markdown(css, unsafe_allow_html=True)

class DashboardComponents:
    """Interactive dashboard components for the app."""
    
    def __init__(self, theme_manager: ThemeManager):
        self.theme_manager = theme_manager
        self.colors = theme_manager.get_theme_colors()
    
    def create_verification_score_chart(self, score: float, breakdown: Dict[str, float]) -> go.Figure:
        """Create verification score gauge chart."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            title={'text': "Verification Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': self.colors['accent_color']},
                'steps': [
                    {'range': [0, 40], 'color': "#ff6b6b"},
                    {'range': [40, 70], 'color': "#feca57"},
                    {'range': [70, 100], 'color': "#00D4AA"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': self.colors['text_primary'], 'family': "Inter"},
            height=300
        )
        
        return fig
    
    def create_source_breakdown_chart(self, source_data: Dict[str, int]) -> go.Figure:
        """Create source breakdown pie chart."""
        labels = list(source_data.keys())
        values = list(source_data.values())
        
        colors = ['#667eea', '#764ba2', '#00D4AA', '#48CAE4', '#feca57']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker=dict(colors=colors[:len(labels)]),
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Source Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': self.colors['text_primary'], 'family': "Inter"},
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_confidence_timeline(self, analysis_history: List[Dict[str, Any]]) -> go.Figure:
        """Create confidence score timeline."""
        if not analysis_history:
            # Create sample data for demo
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            scores = np.random.beta(2, 2, 30) * 100  # Generate realistic confidence scores
            analysis_history = [
                {"timestamp": date, "confidence": score}
                for date, score in zip(dates, scores)
            ]
        
        df = pd.DataFrame(analysis_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['confidence'],
            mode='lines+markers',
            name='Confidence Score',
            line=dict(color=self.colors['accent_color'], width=3),
            marker=dict(size=6, color=self.colors['accent_color'])
        ))
        
        fig.update_layout(
            title="Analysis Confidence Over Time",
            xaxis_title="Date",
            yaxis_title="Confidence Score (%)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': self.colors['text_primary'], 'family': "Inter"},
            height=400,
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    
    def create_feature_importance_chart(self, features: List[Tuple[str, float]]) -> go.Figure:
        """Create feature importance horizontal bar chart."""
        if not features:
            return None
        
        # Sort by importance
        features = sorted(features, key=lambda x: abs(x[1]), reverse=True)
        words, importance = zip(*features[:10])  # Top 10 features
        
        # Color based on positive/negative importance
        colors = ['#00D4AA' if imp > 0 else '#ff6b6b' for imp in importance]
        
        fig = go.Figure(go.Bar(
            x=importance,
            y=words,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{imp:.3f}' for imp in importance],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Word Importance for Classification",
            xaxis_title="Importance Score",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': self.colors['text_primary'], 'family': "Inter"},
            height=400,
            margin=dict(l=100)
        )
        
        return fig
    
    def create_suspicious_patterns_radar(self, patterns: Dict[str, bool]) -> go.Figure:
        """Create radar chart for suspicious patterns."""
        if not patterns:
            return None
        
        categories = list(patterns.keys())
        values = [1 if patterns[cat] else 0 for cat in categories]
        
        # Close the radar chart
        categories.append(categories[0])
        values.append(values[0])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(255, 107, 107, 0.3)',
            line=dict(color='#ff6b6b', width=2),
            name='Detected Patterns'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0, 1],
                    ticktext=['Not Detected', 'Detected']
                )
            ),
            title="Suspicious Pattern Detection",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': self.colors['text_primary'], 'family': "Inter"},
            height=500
        )
        
        return fig
    
    def create_similarity_heatmap(self, similarity_matrix: np.ndarray, labels: List[str]) -> go.Figure:
        """Create similarity heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale='RdYlBu_r',
            text=np.round(similarity_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='%{x} vs %{y}<br>Similarity: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Content Similarity Matrix",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': self.colors['text_primary'], 'family': "Inter"},
            height=500
        )
        
        return fig
    
    def display_analysis_summary(self, analysis_results: Dict[str, Any]):
        """Display comprehensive analysis summary."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            confidence = analysis_results.get('classification', {}).get('confidence', 0)
            st.metric(
                "AI Confidence",
                f"{confidence * 100:.1f}%",
                delta=f"{(confidence - 0.5) * 100:.1f}%" if confidence != 0 else None
            )
        
        with col2:
            word_count = analysis_results.get('text_stats', {}).get('word_count', 0)
            st.metric("Word Count", f"{word_count:,}")
        
        with col3:
            patterns_detected = sum(analysis_results.get('suspicious_patterns', {}).values())
            st.metric("Suspicious Patterns", patterns_detected)
        
        with col4:
            verification_score = analysis_results.get('verification_score', 0)
            st.metric(
                "Verification Score", 
                f"{verification_score * 100:.1f}%",
                delta=f"{(verification_score - 0.5) * 100:.1f}%" if verification_score != 0 else None
            )

class LoadingAnimations:
    """Animated loading components."""
    
    @staticmethod
    def show_analysis_progress(steps: List[str]):
        """Show step-by-step analysis progress."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, step in enumerate(steps):
            progress = (i + 1) / len(steps)
            progress_bar.progress(progress)
            status_text.text(f"🔍 {step}...")
            
            # Simulate processing time
            import time
            time.sleep(0.5)
        
        status_text.text("✅ Analysis complete!")
        return progress_bar, status_text
    
    @staticmethod
    def show_loading_spinner(message: str = "Processing..."):
        """Show loading spinner with message."""
        return st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <div class="loading-spinner" style="display: inline-block; font-size: 2rem;">🔄</div>
            <p style="margin-top: 1rem;">{message}</p>
        </div>
        """, unsafe_allow_html=True)

def create_theme_toggle_button(theme_manager: ThemeManager):
    """Create theme toggle button in sidebar."""
    current_theme = theme_manager.current_theme
    
    # Create button in top right
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        theme_icon = "🌙" if current_theme == "light" else "☀️"
        new_theme_name = "Dark" if current_theme == "light" else "Light"
        
        if st.button(f"{theme_icon}", help=f"Switch to {new_theme_name} Theme", key="theme_toggle"):
            theme_manager.toggle_theme()
            st.experimental_rerun()

def display_header(title: str, subtitle: str = ""):
    """Display premium header with animation."""
    st.markdown(f'<div class="main-header fade-in">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="main-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def display_section_header(title: str):
    """Display section header with styling."""
    st.markdown(f'<div class="section-header fade-in">{title}</div>', unsafe_allow_html=True)

def display_info_box(title: str, content: str):
    """Display information box with premium styling."""
    st.markdown(f"""
    <div class="info-box fade-in">
    <h3>{title}</h3>
    {content}
    </div>
    """, unsafe_allow_html=True)

def display_premium_card(title: str, content: str):
    """Display premium card with hover effects."""
    st.markdown(f"""
    <div class="premium-card fade-in">
    <h4>{title}</h4>
    {content}
    </div>
    """, unsafe_allow_html=True)

def display_result_box(verdict_type: str, title: str, content: str):
    """Display result box with appropriate styling."""
    css_class = f"{verdict_type}-result"
    st.markdown(f"""
    <div class="result-box {css_class} fade-in">
    <h3>{title}</h3>
    {content}
    </div>
    """, unsafe_allow_html=True)