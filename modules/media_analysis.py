"""
Advanced media analysis module for image and video deepfake detection.
"""

import cv2
import numpy as np
import tempfile
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F

from .config import MODEL_CONFIGS, ANALYSIS_THRESHOLDS, settings
from .utils import cache_result, logger

class MediaAnalyzer:
    """Advanced media analysis with deepfake detection and explainability."""
    
    def __init__(self):
        self.face_detector = None
        self.detector_type = "None"
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize face detection models."""
        with st.spinner("👤 Loading face detection models..."):
            self._load_face_detectors()
    
    def _load_face_detectors(self):
        """Load face detection models with fallbacks."""
        # Try Haar cascades first (most reliable)
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if not face_cascade.empty():
                self.face_detector = face_cascade
                self.detector_type = "Haar"
                st.success("✅ Haar cascade face detector loaded")
                return
                
        except Exception as e:
            logger.warning(f"Haar cascade loading failed: {e}")
        
        # Try YuNet as fallback
        try:
            model_path = "face_detection_yunet_2023mar.onnx"
            yunet_url = MODEL_CONFIGS["face_detection"]["yunet_url"]
            
            if not os.path.exists(model_path):
                st.info("📥 Downloading YuNet face detection model...")
                
                import requests
                response = requests.get(yunet_url, timeout=60, stream=True)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320))
            self.face_detector = detector
            self.detector_type = "YuNet"
            st.success("✅ YuNet face detector loaded")
            
        except Exception as e:
            logger.error(f"YuNet loading failed: {e}")
            st.error("❌ Face detection models failed to load")
    
    def analyze_image(self, image_data: np.ndarray, explain: bool = True) -> Dict[str, Any]:
        """
        Comprehensive image analysis for deepfake detection.
        
        Args:
            image_data: Image as numpy array
            explain: Whether to generate explanations
            
        Returns:
            Analysis results with predictions and explanations
        """
        if image_data is None or len(image_data.shape) < 2:
            return {"error": "Invalid image data"}
        
        results = {
            "image_info": self._get_image_info(image_data),
            "face_detection": {},
            "deepfake_analysis": {},
            "technical_metrics": {},
            "visual_explanations": {}
        }
        
        # Detect faces
        faces, face_info = self._detect_faces(image_data)
        results["face_detection"] = face_info
        
        if faces is not None and len(faces) > 0:
            # Analyze for deepfake indicators
            results["deepfake_analysis"] = self._analyze_deepfake_indicators(image_data, faces)
            
            # Calculate deepfake score
            results["deepfake_score"] = self._calculate_deepfake_score(
                results["deepfake_analysis"], 
                len(faces)
            )
            
            # Generate visual explanations if requested
            if explain:
                results["visual_explanations"] = self._generate_visual_explanations(
                    image_data, faces, results["deepfake_analysis"]
                )
        
        # Technical image metrics
        results["technical_metrics"] = self._calculate_technical_metrics(image_data)
        
        return results
    
    def analyze_video(self, video_path: str, max_frames: int = None) -> Dict[str, Any]:
        """
        Comprehensive video analysis for deepfake detection.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to analyze
            
        Returns:
            Video analysis results
        """
        if max_frames is None:
            max_frames = settings.max_video_frames
        
        results = {
            "video_info": {},
            "frame_extraction": {},
            "temporal_analysis": {},
            "deepfake_analysis": {},
            "consistency_checks": {}
        }
        
        # Extract video information and frames
        video_info = self._extract_video_info(video_path)
        results["video_info"] = video_info
        
        if "error" in video_info:
            return results
        
        # Extract frames for analysis
        frames_data = self._extract_video_frames(video_path, max_frames)
        results["frame_extraction"] = frames_data
        
        if "error" in frames_data:
            return results
        
        # Analyze each frame
        frame_analyses = []
        frames = frames_data["frames"]
        
        for i, frame in enumerate(frames):
            frame_result = self.analyze_image(frame, explain=False)
            frame_result["frame_number"] = i
            frame_analyses.append(frame_result)
        
        # Temporal consistency analysis
        results["temporal_analysis"] = self._analyze_temporal_consistency(frame_analyses)
        
        # Overall video deepfake analysis
        results["deepfake_analysis"] = self._calculate_video_deepfake_score(frame_analyses)
        
        return results
    
    @cache_result("face_detection", ttl=3600)
    def _detect_faces(self, image_array: np.ndarray) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Detect faces in image with comprehensive error handling."""
        try:
            if len(image_array.shape) == 3:
                height, width = image_array.shape[:2]
            else:
                height, width = image_array.shape
            
            if self.detector_type == "YuNet" and self.face_detector is not None:
                self.face_detector.setInputSize((width, height))
                _, faces = self.face_detector.detect(image_array)
                
                face_info = {
                    "detector_used": "YuNet",
                    "faces_detected": len(faces) if faces is not None else 0,
                    "success": faces is not None and len(faces) > 0,
                    "faces_data": faces.tolist() if faces is not None else []
                }
                
                return faces, face_info
                
            elif self.detector_type == "Haar" and self.face_detector is not None:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
                faces = self.face_detector.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                
                face_info = {
                    "detector_used": "Haar",
                    "faces_detected": len(faces),
                    "success": len(faces) > 0,
                    "faces_data": faces.tolist()
                }
                
                return faces, face_info
            else:
                return None, {
                    "detector_used": "None", 
                    "faces_detected": 0, 
                    "success": False,
                    "error": "No face detector available"
                }
                
        except Exception as e:
            return None, {
                "detector_used": self.detector_type, 
                "faces_detected": 0, 
                "success": False,
                "error": str(e)
            }
    
    def _analyze_deepfake_indicators(self, image_array: np.ndarray, faces: Any) -> Dict[str, Any]:
        """Analyze various deepfake indicators in the image."""
        try:
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            indicators = {}
            
            # 1. Image quality analysis
            indicators.update(self._analyze_image_quality(gray))
            
            # 2. Face-specific analysis
            if faces is not None and len(faces) > 0:
                indicators.update(self._analyze_face_regions(gray, faces))
            
            # 3. Frequency domain analysis
            indicators.update(self._analyze_frequency_domain(gray))
            
            # 4. Edge and texture analysis
            indicators.update(self._analyze_edges_and_texture(gray))
            
            # 5. Color analysis (if color image)
            if len(image_array.shape) == 3:
                indicators.update(self._analyze_color_distribution(image_array))
            
            return indicators
            
        except Exception as e:
            return {"error": f"Deepfake analysis failed: {str(e)}"}
    
    def _analyze_image_quality(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Analyze overall image quality indicators."""
        # Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        
        # Brightness and contrast analysis
        mean_brightness = np.mean(gray_image)
        brightness_std = np.std(gray_image)
        
        # Noise analysis
        noise_level = self._estimate_noise_level(gray_image)
        
        return {
            "blur_score": float(laplacian_var),
            "is_blurry": laplacian_var < 100,
            "mean_brightness": float(mean_brightness),
            "brightness_std": float(brightness_std),
            "lighting_inconsistent": brightness_std > 80,
            "noise_level": float(noise_level),
            "high_noise": noise_level > 30
        }
    
    def _analyze_face_regions(self, gray_image: np.ndarray, faces: Any) -> Dict[str, Any]:
        """Analyze face regions for manipulation indicators."""
        face_metrics = []
        
        face_list = faces if isinstance(faces, (list, tuple, np.ndarray)) else [faces]
        
        for face in face_list:
            try:
                if len(face) >= 4:
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    
                    # Ensure coordinates are within bounds
                    x = max(0, min(x, gray_image.shape[1] - 1))
                    y = max(0, min(y, gray_image.shape[0] - 1))
                    w = max(1, min(w, gray_image.shape[1] - x))
                    h = max(1, min(h, gray_image.shape[0] - y))
                    
                    if w > 0 and h > 0:
                        face_roi = gray_image[y:y+h, x:x+w]
                        
                        if face_roi.size > 0:
                            # Face-specific metrics
                            face_blur = cv2.Laplacian(face_roi, cv2.CV_64F).var()
                            face_brightness = np.mean(face_roi)
                            face_contrast = np.std(face_roi)
                            
                            # Symmetry analysis (simplified)
                            left_half = face_roi[:, :w//2]
                            right_half = cv2.flip(face_roi[:, w//2:], 1)
                            
                            if left_half.shape == right_half.shape:
                                symmetry_score = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
                            else:
                                symmetry_score = 0.0
                            
                            face_metrics.append({
                                'blur': float(face_blur),
                                'brightness': float(face_brightness),
                                'contrast': float(face_contrast),
                                'area': w * h,
                                'symmetry': float(symmetry_score) if not np.isnan(symmetry_score) else 0.0
                            })
            except Exception as e:
                logger.warning(f"Face analysis failed for face region: {e}")
                continue
        
        if face_metrics:
            return {
                "face_blur_avg": np.mean([m['blur'] for m in face_metrics]),
                "face_brightness_consistency": np.std([m['brightness'] for m in face_metrics]),
                "face_contrast_consistency": np.std([m['contrast'] for m in face_metrics]),
                "face_size_consistency": np.std([m['area'] for m in face_metrics]),
                "face_symmetry_avg": np.mean([m['symmetry'] for m in face_metrics]),
                "num_faces_analyzed": len(face_metrics)
            }
        else:
            return {"num_faces_analyzed": 0}
    
    def _analyze_frequency_domain(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Analyze frequency domain for manipulation artifacts."""
        try:
            # FFT analysis
            f_transform = np.fft.fft2(gray_image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
            
            rows, cols = gray_image.shape
            crow, ccol = rows // 2, cols // 2
            
            # High frequency energy (potential manipulation indicator)
            high_freq_region = magnitude_spectrum[
                max(0, crow-30):crow+30, 
                max(0, ccol-30):ccol+30
            ]
            high_freq_energy = np.mean(high_freq_region)
            
            # Low frequency dominance
            low_freq_region = magnitude_spectrum[
                max(0, crow-10):crow+10, 
                max(0, ccol-10):ccol+10
            ]
            low_freq_energy = np.mean(low_freq_region)
            
            return {
                "high_freq_energy": float(high_freq_energy),
                "low_freq_energy": float(low_freq_energy),
                "freq_ratio": float(high_freq_energy / low_freq_energy) if low_freq_energy > 0 else 0.0,
                "suspicious_freq_pattern": high_freq_energy > 15
            }
            
        except Exception as e:
            logger.warning(f"Frequency analysis failed: {e}")
            return {"high_freq_energy": 0.0, "freq_analysis_error": True}
    
    def _analyze_edges_and_texture(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Analyze edge patterns and texture consistency."""
        try:
            # Edge detection
            edges = cv2.Canny(gray_image, 50, 150)
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
            
            # Texture analysis using Local Binary Patterns (simplified)
            texture_variance = self._calculate_texture_variance(gray_image)
            
            return {
                "edge_density": float(edge_density),
                "edge_anomaly": edge_density < 0.01 or edge_density > 0.1,
                "texture_variance": float(texture_variance),
                "texture_inconsistency": texture_variance < 100 or texture_variance > 5000
            }
            
        except Exception as e:
            logger.warning(f"Edge analysis failed: {e}")
            return {"edge_density": 0.0, "edge_analysis_error": True}
    
    def _analyze_color_distribution(self, color_image: np.ndarray) -> Dict[str, float]:
        """Analyze color distribution for manipulation signs."""
        try:
            # Split color channels
            b, g, r = cv2.split(color_image)
            
            # Color balance analysis
            r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
            
            # Color consistency
            r_std, g_std, b_std = np.std(r), np.std(g), np.std(b)
            
            # Color temperature estimation (simplified)
            color_temp_ratio = (r_mean + g_mean) / (b_mean + 1)
            
            return {
                "red_dominance": float(r_mean / (r_mean + g_mean + b_mean + 1)),
                "green_dominance": float(g_mean / (r_mean + g_mean + b_mean + 1)),
                "blue_dominance": float(b_mean / (r_mean + g_mean + b_mean + 1)),
                "color_balance_score": float(np.std([r_mean, g_mean, b_mean])),
                "color_consistency": float(np.mean([r_std, g_std, b_std])),
                "color_temp_ratio": float(color_temp_ratio),
                "unnatural_coloring": color_temp_ratio < 0.8 or color_temp_ratio > 2.5
            }
            
        except Exception as e:
            logger.warning(f"Color analysis failed: {e}")
            return {"color_analysis_error": True}
    
    def _calculate_deepfake_score(self, indicators: Dict[str, Any], num_faces: int) -> Dict[str, Any]:
        """Calculate overall deepfake probability score."""
        if "error" in indicators:
            return {"error": indicators["error"], "suspicion_score": 0.5}
        
        suspicion_score = 0.0
        factors = []
        weights = {
            "image_quality": 0.25,
            "face_analysis": 0.35,
            "frequency_domain": 0.20,
            "edge_texture": 0.15,
            "color_analysis": 0.05
        }
        
        # Image quality factors
        if indicators.get('is_blurry'):
            suspicion_score += 0.15 * weights["image_quality"]
            factors.append("Unusual blur patterns")
        
        if indicators.get('lighting_inconsistent'):
            suspicion_score += 0.20 * weights["image_quality"]
            factors.append("Inconsistent lighting")
        
        # Face analysis factors
        if indicators.get('face_blur_avg', 0) < 50 and num_faces > 0:
            suspicion_score += 0.20 * weights["face_analysis"]
            factors.append("Artificially smooth faces")
        
        if indicators.get('face_brightness_consistency', 0) > 30:
            suspicion_score += 0.15 * weights["face_analysis"]
            factors.append("Inconsistent face lighting")
        
        if indicators.get('face_symmetry_avg', 0.8) < 0.3:
            suspicion_score += 0.10 * weights["face_analysis"]
            factors.append("Unusual facial asymmetry")
        
        # Frequency domain factors
        if indicators.get('suspicious_freq_pattern'):
            suspicion_score += 0.25 * weights["frequency_domain"]
            factors.append("Suspicious frequency patterns")
        
        # Edge and texture factors
        if indicators.get('edge_anomaly'):
            suspicion_score += 0.20 * weights["edge_texture"]
            factors.append("Unusual edge patterns")
        
        if indicators.get('texture_inconsistency'):
            suspicion_score += 0.15 * weights["edge_texture"]
            factors.append("Texture inconsistencies")
        
        # Color analysis factors
        if indicators.get('unnatural_coloring'):
            suspicion_score += 0.30 * weights["color_analysis"]
            factors.append("Unnatural color distribution")
        
        # Multiple faces penalty
        if num_faces > 2:
            suspicion_score += 0.10
            factors.append(f"Multiple faces detected ({num_faces})")
        
        # Cap the score
        suspicion_score = min(suspicion_score, 1.0)
        
        # Determine verdict
        thresholds = ANALYSIS_THRESHOLDS["deepfake"]
        if suspicion_score >= thresholds["high_risk"]:
            verdict = "HIGH RISK - LIKELY DEEPFAKE"
            confidence_level = "high"
        elif suspicion_score >= thresholds["medium_risk"]:
            verdict = "MODERATE RISK - SUSPICIOUS"
            confidence_level = "medium"
        else:
            verdict = "LOW RISK - LIKELY AUTHENTIC"
            confidence_level = "low"
        
        return {
            "suspicion_score": float(suspicion_score),
            "confidence_level": confidence_level,
            "verdict": verdict,
            "factors": factors,
            "num_faces": num_faces,
            "technical_details": {
                "quality_score": 1 - indicators.get('blur_score', 100) / 200,
                "consistency_score": 1 - indicators.get('face_brightness_consistency', 0) / 100,
                "authenticity_probability": 1 - suspicion_score
            }
        }
    
    def _generate_visual_explanations(self, image_array: np.ndarray, faces: Any, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visual explanations for deepfake detection."""
        try:
            explanations = {
                "face_regions": [],
                "suspicious_areas": [],
                "quality_heatmap": None,
                "explanation_text": []
            }
            
            # Mark face regions
            face_regions = []
            face_list = faces if isinstance(faces, (list, tuple, np.ndarray)) else [faces]
            
            for i, face in enumerate(face_list):
                if len(face) >= 4:
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    face_regions.append({
                        "id": i,
                        "bbox": [x, y, w, h],
                        "confidence": "detected"
                    })
            
            explanations["face_regions"] = face_regions
            
            # Generate explanation text
            explanation_text = []
            
            if indicators.get('is_blurry'):
                explanation_text.append("Image shows unusual blur patterns that may indicate manipulation")
            
            if indicators.get('lighting_inconsistent'):
                explanation_text.append("Lighting appears inconsistent across the image")
            
            if indicators.get('face_brightness_consistency', 0) > 30:
                explanation_text.append("Face regions show inconsistent lighting patterns")
            
            if indicators.get('suspicious_freq_pattern'):
                explanation_text.append("Frequency analysis reveals potential manipulation artifacts")
            
            if not explanation_text:
                explanation_text.append("No significant manipulation indicators detected")
            
            explanations["explanation_text"] = explanation_text
            
            return explanations
            
        except Exception as e:
            logger.warning(f"Visual explanation generation failed: {e}")
            return {"error": str(e)}
    
    def _get_image_info(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Get basic image information."""
        return {
            "height": image_array.shape[0],
            "width": image_array.shape[1],
            "channels": image_array.shape[2] if len(image_array.shape) > 2 else 1,
            "data_type": str(image_array.dtype),
            "size_bytes": image_array.nbytes
        }
    
    def _calculate_technical_metrics(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive technical metrics."""
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
        
        return {
            "resolution": f"{image_array.shape[1]}x{image_array.shape[0]}",
            "aspect_ratio": image_array.shape[1] / image_array.shape[0],
            "pixel_count": image_array.shape[0] * image_array.shape[1],
            "mean_intensity": float(np.mean(gray)),
            "std_intensity": float(np.std(gray)),
            "min_intensity": int(np.min(gray)),
            "max_intensity": int(np.max(gray)),
            "dynamic_range": int(np.max(gray) - np.min(gray))
        }
    
    def _extract_video_info(self, video_path: str) -> Dict[str, Any]:
        """Extract basic video information."""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {"error": "Could not open video file"}
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                "total_frames": total_frames,
                "fps": float(fps),
                "width": width,
                "height": height,
                "duration": float(duration),
                "resolution": f"{width}x{height}",
                "file_path": video_path
            }
            
        except Exception as e:
            return {"error": f"Video info extraction failed: {str(e)}"}
    
    def _extract_video_frames(self, video_path: str, max_frames: int) -> Dict[str, Any]:
        """Extract frames from video for analysis."""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(1, total_frames // max_frames)
            
            frame_count = 0
            extracted_count = 0
            
            while cap.isOpened() and extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            
            return {
                "frames": frames,
                "extracted_count": extracted_count,
                "total_frames": total_frames,
                "interval": interval
            }
            
        except Exception as e:
            return {"error": f"Frame extraction failed: {str(e)}"}
    
    def _analyze_temporal_consistency(self, frame_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal consistency across video frames."""
        if not frame_analyses:
            return {"error": "No frame analyses provided"}
        
        try:
            # Extract metrics from each frame
            face_counts = []
            brightness_values = []
            blur_scores = []
            suspicion_scores = []
            
            for analysis in frame_analyses:
                face_detection = analysis.get("face_detection", {})
                face_counts.append(face_detection.get("faces_detected", 0))
                
                technical = analysis.get("technical_metrics", {})
                brightness_values.append(technical.get("mean_intensity", 128))
                
                deepfake = analysis.get("deepfake_analysis", {})
                blur_scores.append(deepfake.get("blur_score", 100))
                
                score_data = analysis.get("deepfake_score", {})
                suspicion_scores.append(score_data.get("suspicion_score", 0.0))
            
            # Calculate consistency metrics
            face_consistency = np.std(face_counts) if len(face_counts) > 1 else 0
            brightness_consistency = np.std(brightness_values) if len(brightness_values) > 1 else 0
            blur_consistency = np.std(blur_scores) if len(blur_scores) > 1 else 0
            
            return {
                "face_count_consistency": float(face_consistency),
                "brightness_consistency": float(brightness_consistency),
                "blur_consistency": float(blur_consistency),
                "avg_suspicion_score": float(np.mean(suspicion_scores)),
                "suspicion_variance": float(np.var(suspicion_scores)),
                "temporal_inconsistency_detected": face_consistency > 2.0 or brightness_consistency > 30,
                "frames_analyzed": len(frame_analyses)
            }
            
        except Exception as e:
            return {"error": f"Temporal analysis failed: {str(e)}"}
    
    def _calculate_video_deepfake_score(self, frame_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall video deepfake probability."""
        if not frame_analyses:
            return {"error": "No frame analyses provided"}
        
        try:
            suspicion_scores = []
            suspicious_frames = 0
            
            for analysis in frame_analyses:
                score_data = analysis.get("deepfake_score", {})
                suspicion = score_data.get("suspicion_score", 0.0)
                suspicion_scores.append(suspicion)
                
                if suspicion > 0.5:
                    suspicious_frames += 1
            
            avg_suspicion = np.mean(suspicion_scores)
            max_suspicion = np.max(suspicion_scores)
            suspicion_variance = np.var(suspicion_scores)
            
            # Video-specific factors
            suspicious_ratio = suspicious_frames / len(frame_analyses)
            
            # Adjust score based on video-specific factors
            video_suspicion = avg_suspicion
            
            if suspicious_ratio > 0.7:
                video_suspicion += 0.2
            elif suspicious_ratio > 0.3:
                video_suspicion += 0.1
            
            if suspicion_variance > 0.1:  # High variance indicates inconsistency
                video_suspicion += 0.1
            
            video_suspicion = min(video_suspicion, 1.0)
            
            # Determine verdict
            thresholds = ANALYSIS_THRESHOLDS["deepfake"]
            if video_suspicion >= thresholds["high_risk"]:
                verdict = "HIGH RISK - LIKELY DEEPFAKE VIDEO"
                confidence_level = "high"
            elif video_suspicion >= thresholds["medium_risk"]:
                verdict = "MODERATE RISK - SUSPICIOUS VIDEO"
                confidence_level = "medium"
            else:
                verdict = "LOW RISK - LIKELY AUTHENTIC VIDEO"
                confidence_level = "low"
            
            return {
                "overall_suspicion": float(video_suspicion),
                "average_suspicion": float(avg_suspicion),
                "max_suspicion": float(max_suspicion),
                "suspicious_frames": suspicious_frames,
                "total_frames": len(frame_analyses),
                "suspicious_ratio": float(suspicious_ratio),
                "confidence_level": confidence_level,
                "verdict": verdict,
                "video_specific_factors": {
                    "frame_consistency": suspicion_variance < 0.05,
                    "temporal_stability": suspicious_ratio < 0.3,
                    "quality_consistency": True  # Would need more analysis
                }
            }
            
        except Exception as e:
            return {"error": f"Video scoring failed: {str(e)}"}
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level in image."""
        try:
            # Use Laplacian to estimate noise
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            noise_estimate = laplacian.var()
            return noise_estimate
        except:
            return 0.0
    
    def _calculate_texture_variance(self, image: np.ndarray) -> float:
        """Calculate texture variance (simplified LBP-like measure)."""
        try:
            # Simple texture measure using local variance
            kernel = np.ones((3, 3)) / 9
            local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((image.astype(np.float32) - local_mean) ** 2, -1, kernel)
            return np.mean(local_variance)
        except:
            return 0.0

# Global media analyzer instance
_media_analyzer = None

def get_media_analyzer() -> MediaAnalyzer:
    """Get or create global media analyzer instance."""
    global _media_analyzer
    if _media_analyzer is None:
        _media_analyzer = MediaAnalyzer()
    return _media_analyzer