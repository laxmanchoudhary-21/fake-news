"""
Enhanced media analysis module with dynamic face detection and real-time visualization.
"""

import cv2
import numpy as np
import tempfile
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64

from .config import MODEL_CONFIGS, ANALYSIS_THRESHOLDS, settings
from .utils import cache_result, logger

class EnhancedMediaAnalyzer:
    """Enhanced media analysis with dynamic visualization and real-time face detection."""
    
    def __init__(self):
        self.face_detector = None
        self.detector_type = "None"
        self.detection_confidence_threshold = 0.5
        self.visualization_cache = {}
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize face detection models with enhanced capabilities."""
        with st.spinner("👤 Loading enhanced face detection models..."):
            self._load_enhanced_face_detectors()
    
    def _load_enhanced_face_detectors(self):
        """Load enhanced face detection models with multiple fallbacks."""
        detection_methods = []
        
        # Method 1: Try Haar cascades (most reliable and fast)
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if not face_cascade.empty():
                self.face_detector = face_cascade
                self.detector_type = "Haar"
                detection_methods.append("Haar Cascade")
                st.success("✅ Haar cascade face detector loaded")
                
                # Also load additional cascades
                self.profile_cascade = None
                self.eye_cascade = None
                self.smile_cascade = None
                
                try:
                    profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
                    self.profile_cascade = cv2.CascadeClassifier(profile_path)
                    detection_methods.append("Profile Detection")
                    
                    eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
                    self.eye_cascade = cv2.CascadeClassifier(eye_path)
                    detection_methods.append("Eye Detection")
                    
                    smile_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
                    self.smile_cascade = cv2.CascadeClassifier(smile_path)
                    detection_methods.append("Smile Detection")
                    
                except Exception as e:
                    logger.warning(f"Additional cascades loading failed: {e}")
                
                st.info(f"🎯 Available detection methods: {', '.join(detection_methods)}")
                return
                
        except Exception as e:
            logger.warning(f"Haar cascade loading failed: {e}")
        
        # Method 2: Try YuNet as fallback
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
    
    def analyze_image_with_visualization(self, image_data: np.ndarray, explain: bool = True, show_realtime: bool = True) -> Dict[str, Any]:
        """
        Enhanced image analysis with real-time visualization.
        
        Args:
            image_data: Image as numpy array
            explain: Whether to generate explanations
            show_realtime: Whether to show real-time detection
            
        Returns:
            Enhanced analysis results with visualizations
        """
        if image_data is None or len(image_data.shape) < 2:
            return {"error": "Invalid image data"}
        
        analysis_start = time.time()
        
        results = {
            "image_info": self._get_enhanced_image_info(image_data),
            "face_detection": {},
            "deepfake_analysis": {},
            "technical_metrics": {},
            "visual_explanations": {},
            "real_time_visualizations": {},
            "processing_timeline": []
        }
        
        # Step 1: Enhanced face detection with visualization
        if show_realtime:
            st.write("🔍 **Step 1: Face Detection Analysis**")
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
        status_text.text("🔎 Detecting faces...")
        progress_bar.progress(0.2)
        
        faces, face_info = self._enhanced_face_detection(image_data)
        results["face_detection"] = face_info
        
        # Create face detection visualization
        if faces is not None and len(faces) > 0:
            detection_viz = self._create_face_detection_visualization(image_data, faces, face_info)
            results["real_time_visualizations"]["face_detection"] = detection_viz
            
            if show_realtime:
                st.image(detection_viz["annotated_image"], caption=f"✅ Detected {len(faces)} face(s)", use_column_width=True)
        
        progress_bar.progress(0.4)
        status_text.text("🔬 Analyzing image quality...")
        
        # Step 2: Enhanced deepfake analysis
        if faces is not None and len(faces) > 0:
            deepfake_analysis = self._enhanced_deepfake_analysis(image_data, faces)
            results["deepfake_analysis"] = deepfake_analysis
            
            # Calculate enhanced deepfake score
            results["deepfake_score"] = self._calculate_enhanced_deepfake_score(
                deepfake_analysis, 
                len(faces),
                image_data.shape
            )
            
            progress_bar.progress(0.7)
            status_text.text("🎯 Generating explanations...")
            
            # Step 3: Generate enhanced visual explanations
            if explain:
                visual_explanations = self._generate_enhanced_visual_explanations(
                    image_data, faces, deepfake_analysis, results["deepfake_score"]
                )
                results["visual_explanations"] = visual_explanations
                results["real_time_visualizations"]["explanations"] = visual_explanations
                
                if show_realtime and "explanation_image" in visual_explanations:
                    st.image(visual_explanations["explanation_image"], 
                           caption="🔍 Deepfake Analysis Explanation", use_column_width=True)
        
        progress_bar.progress(0.9)
        status_text.text("📊 Calculating technical metrics...")
        
        # Step 4: Enhanced technical metrics
        results["technical_metrics"] = self._calculate_enhanced_technical_metrics(image_data)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        
        # Processing timeline
        results["processing_timeline"].append({
            "step": "complete_analysis",
            "duration": time.time() - analysis_start,
            "timestamp": datetime.now().isoformat()
        })
        
        return results
    
    def _enhanced_face_detection(self, image_array: np.ndarray) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Enhanced face detection with multiple methods and confidence scoring."""
        try:
            if len(image_array.shape) == 3:
                height, width = image_array.shape[:2]
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                height, width = image_array.shape
                gray = image_array
            
            detection_results = {
                "detector_used": self.detector_type,
                "methods_attempted": [],
                "faces_detected": 0,
                "success": False,
                "detection_confidence": [],
                "face_qualities": [],
                "additional_features": {}
            }
            
            all_faces = []
            
            if self.detector_type == "YuNet" and self.face_detector is not None:
                try:
                    self.face_detector.setInputSize((width, height))
                    _, faces = self.face_detector.detect(image_array)
                    
                    if faces is not None and len(faces) > 0:
                        detection_results["methods_attempted"].append("YuNet")
                        
                        for face in faces:
                            # YuNet returns [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, confidence]
                            if len(face) >= 15:
                                confidence = face[14]
                                if confidence > self.detection_confidence_threshold:
                                    all_faces.append(face)
                                    detection_results["detection_confidence"].append(float(confidence))
                        
                        detection_results["faces_detected"] = len(all_faces)
                        detection_results["success"] = len(all_faces) > 0
                        detection_results["faces_data"] = [face.tolist() for face in all_faces]
                        
                        return all_faces, detection_results
                        
                except Exception as e:
                    logger.warning(f"YuNet detection failed: {e}")
                    detection_results["methods_attempted"].append(f"YuNet (failed: {e})")
                    
            elif self.detector_type == "Haar" and self.face_detector is not None:
                try:
                    # Main face detection
                    faces = self.face_detector.detectMultiScale(
                        gray, 
                        scaleFactor=1.1, 
                        minNeighbors=5, 
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    detection_results["methods_attempted"].append("Haar Frontal")
                    
                    if len(faces) > 0:
                        all_faces = faces.copy()
                        
                        # Calculate confidence scores for Haar (approximation based on size and position)
                        for face in faces:
                            x, y, w, h = face
                            face_area = w * h
                            img_area = width * height
                            area_ratio = face_area / img_area
                            
                            # Confidence based on face size and position
                            confidence = min(0.9, 0.3 + area_ratio * 10)  # Approximation
                            detection_results["detection_confidence"].append(confidence)
                            
                            # Face quality assessment
                            face_roi = gray[y:y+h, x:x+w]
                            if face_roi.size > 0:
                                quality = self._assess_face_quality(face_roi)
                                detection_results["face_qualities"].append(quality)
                        
                        # Try profile detection if available
                        if hasattr(self, 'profile_cascade') and self.profile_cascade is not None:
                            try:
                                profile_faces = self.profile_cascade.detectMultiScale(
                                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                                )
                                if len(profile_faces) > 0:
                                    detection_results["methods_attempted"].append("Haar Profile")
                                    detection_results["additional_features"]["profile_faces"] = len(profile_faces)
                            except Exception as e:
                                logger.warning(f"Profile detection failed: {e}")
                        
                        # Try eye detection if available
                        if hasattr(self, 'eye_cascade') and self.eye_cascade is not None:
                            try:
                                eyes = self.eye_cascade.detectMultiScale(gray)
                                detection_results["additional_features"]["eyes_detected"] = len(eyes)
                                detection_results["methods_attempted"].append("Eye Detection")
                            except Exception as e:
                                logger.warning(f"Eye detection failed: {e}")
                        
                        # Try smile detection if available
                        if hasattr(self, 'smile_cascade') and self.smile_cascade is not None:
                            try:
                                smiles = self.smile_cascade.detectMultiScale(gray)
                                detection_results["additional_features"]["smiles_detected"] = len(smiles)
                                detection_results["methods_attempted"].append("Smile Detection")
                            except Exception as e:
                                logger.warning(f"Smile detection failed: {e}")
                        
                        detection_results["faces_detected"] = len(all_faces)
                        detection_results["success"] = True
                        detection_results["faces_data"] = all_faces.tolist()
                        
                        return all_faces, detection_results
                
                except Exception as e:
                    logger.warning(f"Haar detection failed: {e}")
                    detection_results["methods_attempted"].append(f"Haar (failed: {e})")
            
            # No faces detected
            return None, detection_results
                
        except Exception as e:
            return None, {
                "detector_used": self.detector_type, 
                "faces_detected": 0, 
                "success": False,
                "error": str(e),
                "methods_attempted": []
            }
    
    def _assess_face_quality(self, face_roi: np.ndarray) -> Dict[str, float]:
        """Assess the quality of detected face region."""
        try:
            if face_roi.size == 0:
                return {"overall_quality": 0.0}
            
            # Blur assessment
            blur_score = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            
            # Brightness assessment
            brightness = np.mean(face_roi)
            brightness_score = 1.0 - abs(brightness - 128) / 128  # Optimal around 128
            
            # Contrast assessment
            contrast = np.std(face_roi)
            contrast_score = min(contrast / 50, 1.0)  # Normalize to 0-1
            
            # Size assessment (larger faces generally higher quality)
            size_score = min(face_roi.shape[0] * face_roi.shape[1] / 10000, 1.0)
            
            # Overall quality score
            overall_quality = (blur_score/200 + brightness_score + contrast_score + size_score) / 4
            
            return {
                "overall_quality": min(overall_quality, 1.0),
                "blur_score": float(blur_score),
                "brightness_score": float(brightness_score),
                "contrast_score": float(contrast_score),
                "size_score": float(size_score),
                "face_area": int(face_roi.shape[0] * face_roi.shape[1])
            }
            
        except Exception as e:
            logger.warning(f"Face quality assessment failed: {e}")
            return {"overall_quality": 0.5, "error": str(e)}
    
    def _create_face_detection_visualization(self, image_array: np.ndarray, faces: Any, face_info: Dict) -> Dict[str, Any]:
        """Create enhanced face detection visualization."""
        try:
            # Create a copy of the original image for annotation
            viz_image = image_array.copy()
            
            # Convert to PIL for better drawing capabilities
            pil_image = Image.fromarray(viz_image)
            draw = ImageDraw.Draw(pil_image)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            detection_stats = {
                "faces_annotated": 0,
                "confidence_scores": [],
                "face_sizes": [],
                "colors_used": []
            }
            
            # Define colors for different confidence levels
            confidence_colors = {
                "high": "#00FF00",    # Green
                "medium": "#FFFF00",  # Yellow  
                "low": "#FF6600",     # Orange
                "very_low": "#FF0000" # Red
            }
            
            face_list = faces if isinstance(faces, (list, tuple, np.ndarray)) else [faces]
            
            for i, face in enumerate(face_list):
                try:
                    if self.detector_type == "YuNet" and len(face) >= 15:
                        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                        confidence = face[14]
                        
                        # Facial landmarks
                        landmarks = {
                            "right_eye": (int(face[4]), int(face[5])),
                            "left_eye": (int(face[6]), int(face[7])),
                            "nose": (int(face[8]), int(face[9])),
                            "right_mouth": (int(face[10]), int(face[11])),
                            "left_mouth": (int(face[12]), int(face[13]))
                        }
                        
                    elif self.detector_type == "Haar" and len(face) >= 4:
                        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                        confidence = face_info.get("detection_confidence", [0.8])[i] if i < len(face_info.get("detection_confidence", [])) else 0.8
                        landmarks = None
                    else:
                        continue
                    
                    # Determine confidence level and color
                    if confidence > 0.8:
                        conf_level = "high"
                    elif confidence > 0.6:
                        conf_level = "medium"
                    elif confidence > 0.4:
                        conf_level = "low"
                    else:
                        conf_level = "very_low"
                    
                    color = confidence_colors[conf_level]
                    detection_stats["colors_used"].append(color)
                    detection_stats["confidence_scores"].append(confidence)
                    detection_stats["face_sizes"].append(w * h)
                    
                    # Draw bounding box with thickness based on confidence
                    thickness = max(2, int(confidence * 6))
                    
                    # Draw rectangle
                    draw.rectangle(
                        [x, y, x + w, y + h],
                        outline=color,
                        width=thickness
                    )
                    
                    # Draw confidence label
                    label = f"Face {i+1}: {confidence:.2f}"
                    
                    # Calculate label background size
                    if font:
                        bbox = draw.textbbox((0, 0), label, font=font)
                        label_width = bbox[2] - bbox[0]
                        label_height = bbox[3] - bbox[1]
                    else:
                        label_width = len(label) * 10
                        label_height = 20
                    
                    # Draw label background
                    label_bg_coords = [
                        x, y - label_height - 5,
                        x + label_width + 10, y
                    ]
                    draw.rectangle(label_bg_coords, fill=color, outline=color)
                    
                    # Draw label text
                    if font:
                        draw.text((x + 5, y - label_height - 2), label, fill="black", font=font)
                    else:
                        draw.text((x + 5, y - 18), label, fill="black")
                    
                    # Draw landmarks if available
                    if landmarks and self.detector_type == "YuNet":
                        for landmark_name, (lx, ly) in landmarks.items():
                            if 0 <= lx < viz_image.shape[1] and 0 <= ly < viz_image.shape[0]:
                                # Draw landmark points
                                draw.ellipse([lx-3, ly-3, lx+3, ly+3], fill="#FF0000", outline="#FFFFFF")
                    
                    detection_stats["faces_annotated"] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to annotate face {i}: {e}")
                    continue
            
            # Add detection summary text
            summary_text = f"Detection: {face_info['detector_used']} | Faces: {detection_stats['faces_annotated']}"
            if detection_stats["confidence_scores"]:
                avg_conf = np.mean(detection_stats["confidence_scores"])
                summary_text += f" | Avg Confidence: {avg_conf:.2f}"
            
            # Draw summary at the top
            if font:
                bbox = draw.textbbox((0, 0), summary_text, font=font)
                summary_width = bbox[2] - bbox[0]
                summary_height = bbox[3] - bbox[1]
            else:
                summary_width = len(summary_text) * 8
                summary_height = 16
            
            # Summary background
            draw.rectangle([10, 10, 20 + summary_width, 15 + summary_height], 
                         fill="rgba(0,0,0,0.7)", outline="#FFFFFF")
            
            # Summary text
            if font:
                draw.text((15, 12), summary_text, fill="#FFFFFF", font=font)
            else:
                draw.text((15, 12), summary_text, fill="#FFFFFF")
            
            # Convert back to numpy array
            annotated_image = np.array(pil_image)
            
            return {
                "annotated_image": annotated_image,
                "detection_stats": detection_stats,
                "visualization_method": "PIL_enhanced",
                "faces_processed": detection_stats["faces_annotated"]
            }
            
        except Exception as e:
            logger.error(f"Face detection visualization failed: {e}")
            return {
                "error": str(e),
                "annotated_image": image_array,  # Return original image
                "faces_processed": 0
            }
    
    def _enhanced_deepfake_analysis(self, image_array: np.ndarray, faces: Any) -> Dict[str, Any]:
        """Enhanced deepfake analysis with advanced metrics."""
        try:
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "image_quality_metrics": {},
                "face_specific_metrics": {},
                "frequency_analysis": {},
                "texture_analysis": {},
                "color_analysis": {},
                "advanced_metrics": {}
            }
            
            # Enhanced image quality analysis
            analysis["image_quality_metrics"] = self._enhanced_image_quality_analysis(gray)
            
            # Enhanced face-specific analysis
            if faces is not None and len(faces) > 0:
                analysis["face_specific_metrics"] = self._enhanced_face_analysis(gray, faces)
            
            # Enhanced frequency domain analysis
            analysis["frequency_analysis"] = self._enhanced_frequency_analysis(gray)
            
            # Enhanced texture analysis
            analysis["texture_analysis"] = self._enhanced_texture_analysis(gray)
            
            # Enhanced color analysis (if color image)
            if len(image_array.shape) == 3:
                analysis["color_analysis"] = self._enhanced_color_analysis(image_array)
            
            # Advanced deepfake-specific metrics
            analysis["advanced_metrics"] = self._calculate_advanced_deepfake_metrics(image_array, faces)
            
            return analysis
            
        except Exception as e:
            return {"error": f"Enhanced deepfake analysis failed: {str(e)}"}
    
    def _enhanced_image_quality_analysis(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Enhanced image quality analysis."""
        try:
            # Multiple blur detection methods
            laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            sobel_var = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=3).var()
            
            # Gradient magnitude
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Enhanced brightness and contrast
            mean_brightness = np.mean(gray_image)
            brightness_std = np.std(gray_image)
            
            # Histogram analysis
            hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            hist_entropy = -np.sum(hist * np.log2(hist + 1e-7))  # Avoid log(0)
            
            # Dynamic range
            dynamic_range = np.max(gray_image) - np.min(gray_image)
            
            # Enhanced noise estimation
            noise_estimate = self._estimate_noise_enhanced(gray_image)
            
            return {
                "laplacian_variance": float(laplacian_var),
                "sobel_variance": float(sobel_var),
                "gradient_magnitude_mean": float(np.mean(gradient_magnitude)),
                "is_blurry": laplacian_var < 100,
                "blur_severity": "high" if laplacian_var < 50 else "medium" if laplacian_var < 100 else "low",
                "mean_brightness": float(mean_brightness),
                "brightness_std": float(brightness_std),
                "brightness_balance": "good" if 60 <= mean_brightness <= 180 else "poor",
                "contrast_score": float(brightness_std),
                "histogram_entropy": float(hist_entropy),
                "dynamic_range": int(dynamic_range),
                "noise_estimate": noise_estimate,
                "overall_quality": self._calculate_overall_quality(laplacian_var, mean_brightness, brightness_std, noise_estimate)
            }
            
        except Exception as e:
            logger.error(f"Enhanced image quality analysis failed: {e}")
            return {"error": str(e)}
    
    def _enhanced_face_analysis(self, gray_image: np.ndarray, faces: Any) -> Dict[str, Any]:
        """Enhanced face-specific analysis."""
        try:
            face_metrics = []
            face_list = faces if isinstance(faces, (list, tuple, np.ndarray)) else [faces]
            
            for i, face in enumerate(face_list):
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
                                # Enhanced face metrics
                                face_analysis = {
                                    "face_id": i,
                                    "bbox": [x, y, w, h],
                                    "area": w * h,
                                    "aspect_ratio": w / h,
                                    "position": {
                                        "center_x": x + w // 2,
                                        "center_y": y + h // 2,
                                        "relative_x": (x + w // 2) / gray_image.shape[1],
                                        "relative_y": (y + h // 2) / gray_image.shape[0]
                                    }
                                }
                                
                                # Face quality metrics
                                face_blur = cv2.Laplacian(face_roi, cv2.CV_64F).var()
                                face_brightness = np.mean(face_roi)
                                face_contrast = np.std(face_roi)
                                
                                face_analysis["quality_metrics"] = {
                                    "blur_score": float(face_blur),
                                    "brightness": float(face_brightness),
                                    "contrast": float(face_contrast),
                                    "sharpness": float(face_blur / 100),  # Normalized
                                    "is_well_lit": 60 <= face_brightness <= 200
                                }
                                
                                # Enhanced symmetry analysis
                                symmetry_analysis = self._analyze_face_symmetry(face_roi)
                                face_analysis["symmetry"] = symmetry_analysis
                                
                                # Texture analysis within face
                                texture_analysis = self._analyze_face_texture(face_roi)
                                face_analysis["texture"] = texture_analysis
                                
                                # Face authenticity indicators
                                authenticity_indicators = self._analyze_face_authenticity(face_roi)
                                face_analysis["authenticity_indicators"] = authenticity_indicators
                                
                                face_metrics.append(face_analysis)
                                
                except Exception as e:
                    logger.warning(f"Failed to analyze face {i}: {e}")
                    continue
            
            # Cross-face consistency analysis
            consistency_analysis = {}
            if len(face_metrics) > 1:
                consistency_analysis = self._analyze_face_consistency(face_metrics)
            
            return {
                "individual_faces": face_metrics,
                "face_count": len(face_metrics),
                "consistency_analysis": consistency_analysis,
                "overall_face_quality": np.mean([f["quality_metrics"]["sharpness"] for f in face_metrics]) if face_metrics else 0.0
            }
            
        except Exception as e:
            logger.error(f"Enhanced face analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_face_symmetry(self, face_roi: np.ndarray) -> Dict[str, float]:
        """Analyze facial symmetry for authenticity."""
        try:
            h, w = face_roi.shape
            left_half = face_roi[:, :w//2]
            right_half = face_roi[:, w//2:]
            right_half_flipped = cv2.flip(right_half, 1)
            
            # Resize to match if needed
            if left_half.shape != right_half_flipped.shape:
                min_w = min(left_half.shape[1], right_half_flipped.shape[1])
                left_half = left_half[:, :min_w]
                right_half_flipped = right_half_flipped[:, :min_w]
            
            if left_half.size > 0 and right_half_flipped.size > 0 and left_half.shape == right_half_flipped.shape:
                # Correlation coefficient
                corr_coef = np.corrcoef(left_half.flatten(), right_half_flipped.flatten())[0, 1]
                
                # MSE-based symmetry
                mse_symmetry = np.mean((left_half - right_half_flipped) ** 2)
                
                # Structural similarity
                ssim = cv2.matchTemplate(left_half.astype(np.float32), right_half_flipped.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0, 0]
                
                return {
                    "correlation_symmetry": float(corr_coef) if not np.isnan(corr_coef) else 0.0,
                    "mse_symmetry": float(mse_symmetry),
                    "structural_symmetry": float(ssim),
                    "overall_symmetry": float((corr_coef + (1 - mse_symmetry/10000) + ssim) / 3) if not np.isnan(corr_coef) else 0.5
                }
            else:
                return {"overall_symmetry": 0.5, "error": "Shape mismatch"}
                
        except Exception as e:
            logger.warning(f"Face symmetry analysis failed: {e}")
            return {"overall_symmetry": 0.5, "error": str(e)}
    
    def _analyze_face_texture(self, face_roi: np.ndarray) -> Dict[str, float]:
        """Analyze face texture for manipulation signs."""
        try:
            # Local Binary Pattern analysis
            from skimage.feature import local_binary_pattern
            
            # LBP parameters
            radius = 3
            n_points = 8 * radius
            
            lbp = local_binary_pattern(face_roi, n_points, radius, method='uniform')
            
            # Texture uniformity
            texture_uniformity = np.std(lbp)
            
            # Edge density in face
            edges = cv2.Canny(face_roi, 50, 150)
            edge_density = np.sum(edges) / face_roi.size
            
            # Gradient consistency
            grad_x = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_consistency = np.std(gradient_magnitude)
            
            return {
                "lbp_uniformity": float(texture_uniformity),
                "edge_density": float(edge_density),
                "gradient_consistency": float(gradient_consistency),
                "texture_authenticity_score": float(1.0 - min(1.0, abs(texture_uniformity - 50) / 50))
            }
            
        except ImportError:
            # Fallback without scikit-image
            edges = cv2.Canny(face_roi, 50, 150)
            edge_density = np.sum(edges) / face_roi.size
            
            return {
                "edge_density": float(edge_density),
                "texture_authenticity_score": 0.7  # Default score
            }
            
        except Exception as e:
            logger.warning(f"Face texture analysis failed: {e}")
            return {"texture_authenticity_score": 0.5, "error": str(e)}
    
    def _analyze_face_authenticity(self, face_roi: np.ndarray) -> Dict[str, Any]:
        """Analyze face for authenticity indicators."""
        try:
            indicators = {
                "skin_texture_natural": True,
                "lighting_consistent": True,
                "compression_artifacts": False,
                "unnatural_smoothing": False,
                "pixel_inconsistencies": False
            }
            
            # Skin texture analysis
            texture_variance = np.var(face_roi)
            if texture_variance < 100:  # Too smooth
                indicators["unnatural_smoothing"] = True
                indicators["skin_texture_natural"] = False
            
            # Lighting consistency
            brightness_gradient = np.gradient(face_roi.astype(float))
            gradient_variance = np.var(brightness_gradient)
            if gradient_variance > 1000:  # Inconsistent lighting
                indicators["lighting_consistent"] = False
            
            # Compression artifact detection
            # Look for blocking artifacts (8x8 DCT blocks)
            h, w = face_roi.shape
            block_size = 8
            block_variances = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = face_roi[i:i+block_size, j:j+block_size]
                    block_variances.append(np.var(block))
            
            if len(block_variances) > 0:
                block_var_std = np.std(block_variances)
                if block_var_std > 500:  # High variance in block consistency
                    indicators["compression_artifacts"] = True
            
            # Calculate overall authenticity score
            authentic_count = sum([
                indicators["skin_texture_natural"],
                indicators["lighting_consistent"],
                not indicators["compression_artifacts"],
                not indicators["unnatural_smoothing"],
                not indicators["pixel_inconsistencies"]
            ])
            
            authenticity_score = authentic_count / 5
            
            return {
                "indicators": indicators,
                "authenticity_score": authenticity_score,
                "texture_variance": float(texture_variance),
                "gradient_variance": float(gradient_variance)
            }
            
        except Exception as e:
            logger.warning(f"Face authenticity analysis failed: {e}")
            return {"authenticity_score": 0.5, "error": str(e)}
    
    def _analyze_face_consistency(self, face_metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze consistency across multiple faces."""
        try:
            if len(face_metrics) < 2:
                return {"consistency_score": 1.0, "note": "Single face detected"}
            
            # Extract metrics for comparison
            brightness_values = [f["quality_metrics"]["brightness"] for f in face_metrics]
            contrast_values = [f["quality_metrics"]["contrast"] for f in face_metrics]
            blur_scores = [f["quality_metrics"]["blur_score"] for f in face_metrics]
            
            # Calculate consistency
            brightness_consistency = 1.0 - (np.std(brightness_values) / 50)  # Normalize
            contrast_consistency = 1.0 - (np.std(contrast_values) / 30)
            blur_consistency = 1.0 - (np.std(blur_scores) / 100)
            
            # Overall consistency
            overall_consistency = np.mean([
                max(0, brightness_consistency),
                max(0, contrast_consistency), 
                max(0, blur_consistency)
            ])
            
            return {
                "consistency_score": float(overall_consistency),
                "brightness_consistency": float(max(0, brightness_consistency)),
                "contrast_consistency": float(max(0, contrast_consistency)),
                "blur_consistency": float(max(0, blur_consistency)),
                "faces_compared": len(face_metrics)
            }
            
        except Exception as e:
            logger.warning(f"Face consistency analysis failed: {e}")
            return {"consistency_score": 0.5, "error": str(e)}
    
    def _enhanced_frequency_analysis(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Enhanced frequency domain analysis."""
        try:
            # FFT analysis
            f_transform = np.fft.fft2(gray_image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
            
            rows, cols = gray_image.shape
            crow, ccol = rows // 2, cols // 2
            
            # Define frequency regions
            low_freq_mask = np.zeros((rows, cols), dtype=np.uint8)
            cv2.circle(low_freq_mask, (ccol, crow), 30, 1, -1)
            
            high_freq_mask = np.ones((rows, cols), dtype=np.uint8)
            cv2.circle(high_freq_mask, (ccol, crow), 30, 0, -1)
            
            mid_freq_mask = np.zeros((rows, cols), dtype=np.uint8)
            cv2.circle(mid_freq_mask, (ccol, crow), 100, 1, -1)
            cv2.circle(mid_freq_mask, (ccol, crow), 30, 0, -1)
            
            # Calculate energy in different frequency bands
            low_freq_energy = np.mean(magnitude_spectrum[low_freq_mask == 1])
            mid_freq_energy = np.mean(magnitude_spectrum[mid_freq_mask == 1])
            high_freq_energy = np.mean(magnitude_spectrum[high_freq_mask == 1])
            
            # Calculate frequency ratios
            hf_lf_ratio = high_freq_energy / low_freq_energy if low_freq_energy > 0 else 0
            mf_lf_ratio = mid_freq_energy / low_freq_energy if low_freq_energy > 0 else 0
            
            # Detect suspicious frequency patterns
            suspicious_patterns = {
                "excessive_high_freq": high_freq_energy > 15,
                "unusual_freq_ratio": hf_lf_ratio > 0.3 or hf_lf_ratio < 0.05,
                "frequency_peaks": self._detect_frequency_peaks(magnitude_spectrum),
                "compression_artifacts": self._detect_compression_artifacts_freq(magnitude_spectrum)
            }
            
            return {
                "low_freq_energy": float(low_freq_energy),
                "mid_freq_energy": float(mid_freq_energy),
                "high_freq_energy": float(high_freq_energy),
                "hf_lf_ratio": float(hf_lf_ratio),
                "mf_lf_ratio": float(mf_lf_ratio),
                "suspicious_patterns": suspicious_patterns,
                "frequency_authenticity_score": self._calculate_frequency_authenticity(suspicious_patterns)
            }
            
        except Exception as e:
            logger.warning(f"Enhanced frequency analysis failed: {e}")
            return {"frequency_authenticity_score": 0.5, "error": str(e)}
    
    def _detect_frequency_peaks(self, magnitude_spectrum: np.ndarray) -> bool:
        """Detect unusual frequency peaks that might indicate manipulation."""
        try:
            # Find peaks in the magnitude spectrum
            from scipy.signal import find_peaks
            
            # Flatten and find peaks
            flattened = magnitude_spectrum.flatten()
            peaks, _ = find_peaks(flattened, height=np.mean(flattened) + 2 * np.std(flattened))
            
            # If too many peaks, might indicate artifacts
            peak_ratio = len(peaks) / len(flattened)
            return peak_ratio > 0.01  # More than 1% peaks is suspicious
            
        except ImportError:
            # Fallback without scipy
            std_spectrum = np.std(magnitude_spectrum)
            mean_spectrum = np.mean(magnitude_spectrum)
            high_values = magnitude_spectrum > (mean_spectrum + 2 * std_spectrum)
            return np.sum(high_values) / magnitude_spectrum.size > 0.01
            
        except Exception as e:
            logger.warning(f"Frequency peak detection failed: {e}")
            return False
    
    def _detect_compression_artifacts_freq(self, magnitude_spectrum: np.ndarray) -> bool:
        """Detect compression artifacts in frequency domain."""
        try:
            # Look for regular patterns that indicate DCT block compression
            h, w = magnitude_spectrum.shape
            
            # Check for periodic patterns (8x8 blocks)
            block_size = 8
            pattern_strength = 0
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = magnitude_spectrum[i:i+block_size, j:j+block_size]
                    block_mean = np.mean(block)
                    overall_mean = np.mean(magnitude_spectrum)
                    
                    # Check if block has significantly different characteristics
                    if abs(block_mean - overall_mean) > np.std(magnitude_spectrum):
                        pattern_strength += 1
            
            total_blocks = (h // block_size) * (w // block_size)
            artifact_ratio = pattern_strength / total_blocks if total_blocks > 0 else 0
            
            return artifact_ratio > 0.3  # 30% of blocks showing artifacts
            
        except Exception as e:
            logger.warning(f"Compression artifact detection failed: {e}")
            return False
    
    def _calculate_frequency_authenticity(self, suspicious_patterns: Dict[str, bool]) -> float:
        """Calculate authenticity score based on frequency analysis."""
        try:
            suspicious_count = sum(suspicious_patterns.values())
            total_checks = len(suspicious_patterns)
            
            authenticity_score = 1.0 - (suspicious_count / total_checks)
            return max(0.0, authenticity_score)
            
        except Exception as e:
            return 0.5
    
    def _enhanced_texture_analysis(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Enhanced texture analysis for manipulation detection."""
        try:
            # Multiple texture measures
            texture_metrics = {}
            
            # Gradient-based texture
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            texture_metrics["gradient_variance"] = float(np.var(gradient_magnitude))
            texture_metrics["gradient_mean"] = float(np.mean(gradient_magnitude))
            
            # Edge-based texture
            edges = cv2.Canny(gray_image, 50, 150)
            texture_metrics["edge_density"] = float(np.sum(edges) / gray_image.size)
            
            # Variance-based texture (local variance)
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray_image.astype(np.float32) - local_mean) ** 2, -1, kernel)
            texture_metrics["local_variance_mean"] = float(np.mean(local_variance))
            texture_metrics["local_variance_std"] = float(np.std(local_variance))
            
            # Texture uniformity
            texture_metrics["texture_uniformity"] = float(np.std(local_variance))
            
            # Calculate texture authenticity indicators
            authenticity_indicators = {
                "natural_texture_variation": texture_metrics["texture_uniformity"] > 50,
                "appropriate_edge_density": 0.01 < texture_metrics["edge_density"] < 0.1,
                "gradient_consistency": texture_metrics["gradient_variance"] < 10000,
                "local_variation_natural": 100 < texture_metrics["local_variance_mean"] < 2000
            }
            
            # Overall texture authenticity score
            authentic_count = sum(authenticity_indicators.values())
            texture_authenticity_score = authentic_count / len(authenticity_indicators)
            
            return {
                "texture_metrics": texture_metrics,
                "authenticity_indicators": authenticity_indicators,
                "texture_authenticity_score": float(texture_authenticity_score)
            }
            
        except Exception as e:
            logger.warning(f"Enhanced texture analysis failed: {e}")
            return {"texture_authenticity_score": 0.5, "error": str(e)}
    
    def _enhanced_color_analysis(self, color_image: np.ndarray) -> Dict[str, Any]:
        """Enhanced color analysis for manipulation detection."""
        try:
            # Convert to different color spaces for analysis
            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
            lab_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2LAB)
            
            # RGB channel analysis
            r, g, b = cv2.split(color_image)
            
            color_metrics = {
                "rgb_channel_stats": {
                    "red": {"mean": float(np.mean(r)), "std": float(np.std(r))},
                    "green": {"mean": float(np.mean(g)), "std": float(np.std(g))},
                    "blue": {"mean": float(np.mean(b)), "std": float(np.std(b))}
                }
            }
            
            # Color balance analysis
            r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
            color_balance_variance = np.var([r_mean, g_mean, b_mean])
            color_metrics["color_balance_variance"] = float(color_balance_variance)
            
            # HSV analysis
            h, s, v = cv2.split(hsv_image)
            color_metrics["hsv_stats"] = {
                "hue_distribution": float(np.std(h[s > 50])) if np.any(s > 50) else 0.0,  # Only consider saturated pixels
                "saturation_mean": float(np.mean(s)),
                "value_mean": float(np.mean(v))
            }
            
            # LAB analysis
            l, a, b_lab = cv2.split(lab_image)
            color_metrics["lab_stats"] = {
                "lightness_variance": float(np.var(l)),
                "a_channel_variance": float(np.var(a)),
                "b_channel_variance": float(np.var(b_lab))
            }
            
            # Color authenticity indicators
            authenticity_indicators = {
                "natural_color_balance": color_balance_variance < 2000,
                "appropriate_saturation": 30 < color_metrics["hsv_stats"]["saturation_mean"] < 150,
                "natural_hue_distribution": color_metrics["hsv_stats"]["hue_distribution"] > 10,
                "consistent_lightness": color_metrics["lab_stats"]["lightness_variance"] < 3000
            }
            
            # Calculate color authenticity score
            authentic_count = sum(authenticity_indicators.values())
            color_authenticity_score = authentic_count / len(authenticity_indicators)
            
            return {
                "color_metrics": color_metrics,
                "authenticity_indicators": authenticity_indicators,
                "color_authenticity_score": float(color_authenticity_score)
            }
            
        except Exception as e:
            logger.warning(f"Enhanced color analysis failed: {e}")
            return {"color_authenticity_score": 0.5, "error": str(e)}
    
    def _calculate_advanced_deepfake_metrics(self, image_array: np.ndarray, faces: Any) -> Dict[str, Any]:
        """Calculate advanced deepfake-specific metrics."""
        try:
            advanced_metrics = {
                "timestamp": datetime.now().isoformat(),
                "processing_method": "enhanced_analysis"
            }
            
            # Image resolution analysis
            h, w = image_array.shape[:2]
            resolution_score = min(1.0, (h * w) / (1920 * 1080))  # Normalize to Full HD
            advanced_metrics["resolution_authenticity"] = resolution_score
            
            # Aspect ratio analysis
            aspect_ratio = w / h
            natural_ratios = [16/9, 4/3, 3/2, 1/1]  # Common natural ratios
            ratio_naturalness = max([1.0 - abs(aspect_ratio - ratio) for ratio in natural_ratios])
            advanced_metrics["aspect_ratio_naturalness"] = float(ratio_naturalness)
            
            # Compression analysis
            if len(image_array.shape) == 3:
                # Estimate JPEG compression quality
                compression_estimate = self._estimate_compression_quality(image_array)
                advanced_metrics["compression_analysis"] = compression_estimate
            
            # Metadata consistency (would be enhanced with actual EXIF data)
            advanced_metrics["metadata_consistency"] = {
                "has_natural_dimensions": h >= 480 and w >= 640,
                "reasonable_file_size": True,  # Would calculate from actual file
                "dimension_consistency": abs(h - w) < max(h, w)  # Not perfectly square
            }
            
            # Face-to-image ratio analysis
            if faces is not None and len(faces) > 0:
                total_face_area = 0
                face_list = faces if isinstance(faces, (list, tuple, np.ndarray)) else [faces]
                
                for face in face_list:
                    if len(face) >= 4:
                        face_w, face_h = face[2], face[3]
                        total_face_area += face_w * face_h
                
                face_ratio = total_face_area / (h * w)
                advanced_metrics["face_to_image_ratio"] = float(face_ratio)
                advanced_metrics["face_ratio_natural"] = 0.05 <= face_ratio <= 0.7
            
            return advanced_metrics
            
        except Exception as e:
            logger.warning(f"Advanced deepfake metrics calculation failed: {e}")
            return {"error": str(e)}
    
    def _estimate_compression_quality(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Estimate JPEG compression quality."""
        try:
            # Convert to grayscale for analysis
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Look for blocking artifacts (8x8 DCT blocks)
            h, w = gray.shape
            block_size = 8
            block_differences = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    
                    # Check horizontal and vertical discontinuities
                    if i + block_size < h:
                        below_block = gray[i+block_size:i+2*block_size, j:j+block_size]
                        if below_block.shape == block.shape:
                            h_diff = np.mean(np.abs(block[-1, :] - below_block[0, :]))
                            block_differences.append(h_diff)
                    
                    if j + block_size < w:
                        right_block = gray[i:i+block_size, j+block_size:j+2*block_size]
                        if right_block.shape == block.shape:
                            v_diff = np.mean(np.abs(block[:, -1] - right_block[:, 0]))
                            block_differences.append(v_diff)
            
            if block_differences:
                avg_block_diff = np.mean(block_differences)
                
                # Estimate quality (inverse relationship with artifacts)
                if avg_block_diff < 5:
                    estimated_quality = "high"
                    quality_score = 0.9
                elif avg_block_diff < 15:
                    estimated_quality = "medium"
                    quality_score = 0.6
                else:
                    estimated_quality = "low"
                    quality_score = 0.3
            else:
                estimated_quality = "unknown"
                quality_score = 0.5
            
            return {
                "estimated_quality": estimated_quality,
                "quality_score": quality_score,
                "avg_block_discontinuity": float(np.mean(block_differences)) if block_differences else 0.0,
                "compression_artifacts_detected": quality_score < 0.7
            }
            
        except Exception as e:
            logger.warning(f"Compression quality estimation failed: {e}")
            return {"estimated_quality": "unknown", "quality_score": 0.5}
    
    def _calculate_overall_quality(self, blur_score: float, brightness: float, contrast: float, noise: float) -> float:
        """Calculate overall image quality score."""
        try:
            # Normalize components to 0-1 scale
            blur_quality = min(1.0, blur_score / 200)  # Higher blur score = better quality
            brightness_quality = 1.0 - abs(brightness - 128) / 128  # Closer to 128 = better
            contrast_quality = min(1.0, contrast / 50)  # Higher contrast = better (up to a point)
            noise_quality = max(0.0, 1.0 - noise / 100)  # Lower noise = better quality
            
            # Weighted combination
            overall_quality = (
                0.3 * blur_quality +
                0.25 * brightness_quality +
                0.25 * contrast_quality +
                0.2 * noise_quality
            )
            
            return min(1.0, max(0.0, overall_quality))
            
        except Exception as e:
            return 0.5
    
    def _estimate_noise_enhanced(self, image: np.ndarray) -> float:
        """Enhanced noise estimation."""
        try:
            # Method 1: Laplacian-based noise estimation
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            noise_laplacian = laplacian.var()
            
            # Method 2: Median filter based
            median_filtered = cv2.medianBlur(image, 5)
            noise_median = np.mean(np.abs(image.astype(float) - median_filtered.astype(float)))
            
            # Method 3: High-frequency component
            gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
            high_freq = image.astype(float) - gaussian_blur.astype(float)
            noise_hf = np.std(high_freq)
            
            # Combine estimates
            combined_noise = (noise_laplacian/10 + noise_median + noise_hf) / 3
            
            return float(combined_noise)
            
        except Exception as e:
            logger.warning(f"Enhanced noise estimation failed: {e}")
            return 0.0
    
    def _calculate_enhanced_deepfake_score(self, analysis: Dict[str, Any], num_faces: int, image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Calculate enhanced deepfake probability score."""
        if "error" in analysis:
            return {"error": analysis["error"], "suspicion_score": 0.5}
        
        try:
            # Initialize scoring
            suspicion_factors = []
            evidence_details = []
            
            # Image quality factors (25% weight)
            quality_metrics = analysis.get("image_quality_metrics", {})
            quality_suspicion = 0.0
            
            if quality_metrics.get("is_blurry"):
                quality_suspicion += 0.3
                evidence_details.append("Unusual blur patterns detected")
            
            if quality_metrics.get("brightness_balance") == "poor":
                quality_suspicion += 0.2
                evidence_details.append("Poor brightness balance")
            
            if quality_metrics.get("overall_quality", 0.5) < 0.3:
                quality_suspicion += 0.25
                evidence_details.append("Low overall image quality")
            
            suspicion_factors.append(("image_quality", quality_suspicion, 0.25))
            
            # Face analysis factors (35% weight)
            face_metrics = analysis.get("face_specific_metrics", {})
            face_suspicion = 0.0
            
            individual_faces = face_metrics.get("individual_faces", [])
            if individual_faces:
                # Check individual face quality
                poor_quality_faces = 0
                for face in individual_faces:
                    face_quality = face.get("quality_metrics", {})
                    if face_quality.get("sharpness", 1.0) < 0.3:
                        poor_quality_faces += 1
                    
                    # Check authenticity indicators
                    auth_indicators = face.get("authenticity_indicators", {})
                    if auth_indicators.get("authenticity_score", 0.8) < 0.5:
                        poor_quality_faces += 0.5
                
                if poor_quality_faces / len(individual_faces) > 0.5:
                    face_suspicion += 0.4
                    evidence_details.append("Multiple faces show poor quality")
                
                # Check face consistency
                consistency = face_metrics.get("consistency_analysis", {})
                if consistency.get("consistency_score", 1.0) < 0.6:
                    face_suspicion += 0.3
                    evidence_details.append("Inconsistent lighting/quality across faces")
            
            suspicion_factors.append(("face_analysis", face_suspicion, 0.35))
            
            # Frequency analysis factors (20% weight)
            freq_analysis = analysis.get("frequency_analysis", {})
            freq_suspicion = 1.0 - freq_analysis.get("frequency_authenticity_score", 0.7)
            
            suspicious_patterns = freq_analysis.get("suspicious_patterns", {})
            if suspicious_patterns.get("excessive_high_freq"):
                evidence_details.append("Excessive high-frequency content")
            if suspicious_patterns.get("compression_artifacts"):
                evidence_details.append("Compression artifacts in frequency domain")
            if suspicious_patterns.get("frequency_peaks"):
                evidence_details.append("Unusual frequency peaks detected")
            
            suspicion_factors.append(("frequency_analysis", freq_suspicion, 0.20))
            
            # Texture analysis factors (15% weight)
            texture_analysis = analysis.get("texture_analysis", {})
            texture_suspicion = 1.0 - texture_analysis.get("texture_authenticity_score", 0.7)
            
            texture_indicators = texture_analysis.get("authenticity_indicators", {})
            if not texture_indicators.get("natural_texture_variation", True):
                evidence_details.append("Unnatural texture variation")
            if not texture_indicators.get("appropriate_edge_density", True):
                evidence_details.append("Inappropriate edge density")
            
            suspicion_factors.append(("texture_analysis", texture_suspicion, 0.15))
            
            # Color analysis factors (5% weight)
            color_analysis = analysis.get("color_analysis", {})
            color_suspicion = 1.0 - color_analysis.get("color_authenticity_score", 0.7)
            
            color_indicators = color_analysis.get("authenticity_indicators", {})
            if not color_indicators.get("natural_color_balance", True):
                evidence_details.append("Unnatural color balance")
            
            suspicion_factors.append(("color_analysis", color_suspicion, 0.05))
            
            # Calculate weighted suspicion score
            total_suspicion = sum(score * weight for _, score, weight in suspicion_factors)
            
            # Additional factors
            if num_faces > 3:
                total_suspicion += 0.1
                evidence_details.append(f"Multiple faces detected ({num_faces})")
            
            # Resolution factors
            h, w = image_shape[:2]
            if h * w > 8000000:  # Very high resolution might indicate generated content
                total_suspicion += 0.05
                evidence_details.append("Unusually high resolution")
            
            # Cap the suspicion score
            total_suspicion = min(1.0, total_suspicion)
            
            # Determine verdict and confidence
            thresholds = ANALYSIS_THRESHOLDS["deepfake"]
            
            if total_suspicion >= thresholds["high_risk"]:
                verdict = "HIGH RISK - LIKELY DEEPFAKE"
                confidence_level = "high"
                risk_level = "high"
            elif total_suspicion >= thresholds["medium_risk"]:
                verdict = "MODERATE RISK - SUSPICIOUS CONTENT"
                confidence_level = "medium" 
                risk_level = "medium"
            elif total_suspicion >= thresholds["low_risk"]:
                verdict = "LOW-MODERATE RISK - SOME CONCERNS"
                confidence_level = "medium"
                risk_level = "low_medium"
            else:
                verdict = "LOW RISK - LIKELY AUTHENTIC"
                confidence_level = "high"
                risk_level = "low"
            
            return {
                "suspicion_score": float(total_suspicion),
                "authenticity_score": float(1.0 - total_suspicion),
                "confidence_level": confidence_level,
                "risk_level": risk_level,
                "verdict": verdict,
                "evidence_details": evidence_details,
                "factor_breakdown": {
                    "image_quality": suspicion_factors[0][1],
                    "face_analysis": suspicion_factors[1][1],
                    "frequency_analysis": suspicion_factors[2][1],
                    "texture_analysis": suspicion_factors[3][1],
                    "color_analysis": suspicion_factors[4][1]
                },
                "technical_summary": {
                    "faces_analyzed": num_faces,
                    "image_resolution": f"{w}x{h}",
                    "analysis_depth": "comprehensive",
                    "processing_timestamp": datetime.now().isoformat()
                },
                "recommendation": self._generate_image_recommendation(total_suspicion, risk_level)
            }
            
        except Exception as e:
            logger.error(f"Enhanced deepfake score calculation failed: {e}")
            return {"error": str(e), "suspicion_score": 0.5}
    
    def _generate_image_recommendation(self, suspicion_score: float, risk_level: str) -> Dict[str, str]:
        """Generate recommendation based on deepfake analysis."""
        if risk_level == "high":
            return {
                "action": "DO_NOT_TRUST",
                "color": "danger",
                "icon": "🚨",
                "message": "High probability of manipulation detected. Do not trust or share this image.",
                "details": "Multiple indicators suggest this image has been artificially generated or manipulated."
            }
        elif risk_level == "medium":
            return {
                "action": "EXERCISE_CAUTION",
                "color": "warning", 
                "icon": "⚠️",
                "message": "Moderate risk detected. Verify with additional sources before trusting.",
                "details": "Some suspicious indicators found. Consider the source and context carefully."
            }
        elif risk_level == "low_medium":
            return {
                "action": "VERIFY_SOURCE",
                "color": "info",
                "icon": "🔍", 
                "message": "Some concerns detected. Verify the source and context.",
                "details": "Minor quality issues detected that could indicate processing or compression."
            }
        else:
            return {
                "action": "LIKELY_AUTHENTIC",
                "color": "success",
                "icon": "✅",
                "message": "Image appears authentic with no significant manipulation indicators.",
                "details": "Analysis shows characteristics consistent with authentic photography."
            }
    
    def _generate_enhanced_visual_explanations(self, image_array: np.ndarray, faces: Any, analysis: Dict, score: Dict) -> Dict[str, Any]:
        """Generate enhanced visual explanations with overlay annotations."""
        try:
            explanations = {
                "face_regions": [],
                "suspicious_areas": [],
                "quality_heatmap": None,
                "explanation_text": [],
                "risk_visualization": None
            }
            
            # Create explanation image
            explanation_image = self._create_explanation_overlay(image_array, faces, analysis, score)
            explanations["explanation_image"] = explanation_image
            
            # Generate detailed explanation text
            explanations["explanation_text"] = self._generate_explanation_text(analysis, score)
            
            # Create risk level visualization
            risk_viz = self._create_risk_visualization(score)
            explanations["risk_visualization"] = risk_viz
            
            return explanations
            
        except Exception as e:
            logger.error(f"Enhanced visual explanation generation failed: {e}")
            return {"error": str(e)}
    
    def _create_explanation_overlay(self, image_array: np.ndarray, faces: Any, analysis: Dict, score: Dict) -> np.ndarray:
        """Create overlay image with explanations and annotations."""
        try:
            # Create a copy for annotation
            overlay_image = image_array.copy()
            
            # Convert to PIL for better text rendering
            pil_image = Image.fromarray(overlay_image)
            draw = ImageDraw.Draw(pil_image)
            
            # Try to load font
            try:
                font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
                font_medium = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
                font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
            except:
                font_large = font_medium = font_small = ImageFont.load_default()
            
            # Color scheme based on risk level
            risk_level = score.get("risk_level", "low")
            if risk_level == "high":
                primary_color = "#FF0000"
                bg_color = "rgba(255,0,0,0.1)"
            elif risk_level == "medium":
                primary_color = "#FF6600"
                bg_color = "rgba(255,102,0,0.1)"
            else:
                primary_color = "#00AA00"
                bg_color = "rgba(0,170,0,0.1)"
            
            # Draw face regions with risk indicators
            face_list = faces if isinstance(faces, (list, tuple, np.ndarray)) else [faces] if faces is not None else []
            
            for i, face in enumerate(face_list):
                if len(face) >= 4:
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    
                    # Draw face bounding box
                    thickness = 3 if risk_level == "high" else 2
                    draw.rectangle([x, y, x + w, y + h], outline=primary_color, width=thickness)
                    
                    # Face analysis summary
                    face_metrics = analysis.get("face_specific_metrics", {})
                    individual_faces = face_metrics.get("individual_faces", [])
                    
                    if i < len(individual_faces):
                        face_data = individual_faces[i]
                        quality = face_data.get("quality_metrics", {})
                        auth_score = face_data.get("authenticity_indicators", {}).get("authenticity_score", 0.5)
                        
                        # Face quality indicator
                        quality_text = f"Quality: {auth_score:.2f}"
                        draw.text((x, y - 25), quality_text, fill=primary_color, font=font_small)
                        
                        # Quality-based visual indicators
                        if auth_score < 0.5:
                            # Draw warning indicators around suspicious faces
                            for offset in range(0, 360, 45):
                                angle_rad = np.radians(offset)
                                indicator_x = x + w//2 + int(30 * np.cos(angle_rad))
                                indicator_y = y + h//2 + int(30 * np.sin(angle_rad))
                                draw.text((indicator_x, indicator_y), "⚠", fill="#FF0000", font=font_medium)
            
            # Add overall analysis summary
            verdict = score.get("verdict", "ANALYSIS COMPLETE")
            suspicion_score = score.get("suspicion_score", 0.0)
            
            summary_text = f"{verdict}\nSuspicion Score: {suspicion_score:.2f}"
            
            # Summary background
            text_lines = summary_text.split('\n')
            max_line_width = max([draw.textlength(line, font=font_medium) for line in text_lines])
            text_height = len(text_lines) * 25
            
            summary_bg = [10, 10, 20 + max_line_width, 20 + text_height]
            draw.rectangle(summary_bg, fill=bg_color, outline=primary_color, width=2)
            
            # Summary text
            for i, line in enumerate(text_lines):
                draw.text((15, 15 + i * 25), line, fill=primary_color, font=font_medium)
            
            # Add factor breakdown
            factors = score.get("factor_breakdown", {})
            if factors:
                factor_y = summary_bg[3] + 20
                draw.text((15, factor_y), "Analysis Breakdown:", fill=primary_color, font=font_small)
                
                for j, (factor, value) in enumerate(factors.items()):
                    factor_text = f"• {factor.replace('_', ' ').title()}: {value:.2f}"
                    draw.text((20, factor_y + 20 + j * 15), factor_text, fill=primary_color, font=font_small)
            
            return np.array(pil_image)
            
        except Exception as e:
            logger.error(f"Explanation overlay creation failed: {e}")
            return image_array
    
    def _generate_explanation_text(self, analysis: Dict, score: Dict) -> List[str]:
        """Generate detailed explanation text."""
        explanations = []
        
        try:
            # Overall assessment
            verdict = score.get("verdict", "Unknown")
            suspicion = score.get("suspicion_score", 0.0)
            explanations.append(f"Overall Assessment: {verdict} (Score: {suspicion:.2f})")
            
            # Evidence details
            evidence = score.get("evidence_details", [])
            if evidence:
                explanations.append("Key Findings:")
                explanations.extend([f"• {detail}" for detail in evidence[:5]])  # Top 5
            
            # Technical analysis
            factors = score.get("factor_breakdown", {})
            if factors:
                explanations.append("Analysis Breakdown:")
                for factor, value in factors.items():
                    risk_desc = "High" if value > 0.7 else "Medium" if value > 0.4 else "Low"
                    explanations.append(f"• {factor.replace('_', ' ').title()}: {risk_desc} ({value:.2f})")
            
            # Recommendation
            recommendation = score.get("recommendation", {})
            if recommendation:
                explanations.append(f"Recommendation: {recommendation.get('message', 'No specific recommendation')}")
            
            return explanations
            
        except Exception as e:
            logger.error(f"Explanation text generation failed: {e}")
            return ["Analysis completed with errors"]
    
    def _create_risk_visualization(self, score: Dict) -> Dict[str, Any]:
        """Create risk level visualization."""
        try:
            import matplotlib.pyplot as plt
            import io
            
            # Extract data
            factors = score.get("factor_breakdown", {})
            if not factors:
                return {"error": "No factor data available"}
            
            # Create radar chart
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            # Data preparation
            categories = list(factors.keys())
            values = list(factors.values())
            
            # Close the polygon
            categories += [categories[0]]
            values += [values[0]]
            
            # Plot
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True)
            ax.plot(angles, values, 'o-', linewidth=2, label='Risk Factors')
            ax.fill(angles, values, alpha=0.25)
            
            # Customize
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([cat.replace('_', '\n').title() for cat in categories[:-1]])
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
            ax.grid(True)
            
            plt.title('Deepfake Risk Analysis', size=16, fontweight='bold')
            plt.tight_layout()
            
            # Convert to image
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            # Convert to base64 for embedding
            import base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            plt.close(fig)
            
            return {
                "radar_chart": img_base64,
                "chart_type": "risk_analysis",
                "data_points": len(factors)
            }
            
        except Exception as e:
            logger.error(f"Risk visualization creation failed: {e}")
            return {"error": str(e)}
    
    def _get_enhanced_image_info(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Get enhanced image information."""
        try:
            basic_info = {
                "height": image_array.shape[0],
                "width": image_array.shape[1], 
                "channels": image_array.shape[2] if len(image_array.shape) > 2 else 1,
                "data_type": str(image_array.dtype),
                "size_bytes": image_array.nbytes
            }
            
            # Additional calculations
            basic_info.update({
                "aspect_ratio": basic_info["width"] / basic_info["height"],
                "total_pixels": basic_info["height"] * basic_info["width"],
                "megapixels": (basic_info["height"] * basic_info["width"]) / 1000000,
                "color_space": "RGB" if basic_info["channels"] == 3 else "Grayscale" if basic_info["channels"] == 1 else "Other",
                "bits_per_pixel": basic_info["channels"] * 8 if image_array.dtype == np.uint8 else basic_info["channels"] * 16
            })
            
            return basic_info
            
        except Exception as e:
            logger.error(f"Enhanced image info extraction failed: {e}")
            return {"error": str(e)}
    
    def _calculate_enhanced_technical_metrics(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Calculate enhanced technical metrics."""
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
            
            # Basic metrics
            metrics = {
                "resolution": f"{image_array.shape[1]}x{image_array.shape[0]}",
                "aspect_ratio": image_array.shape[1] / image_array.shape[0],
                "pixel_count": image_array.shape[0] * image_array.shape[1],
                "megapixels": (image_array.shape[0] * image_array.shape[1]) / 1000000
            }
            
            # Intensity statistics
            metrics.update({
                "mean_intensity": float(np.mean(gray)),
                "std_intensity": float(np.std(gray)),
                "min_intensity": int(np.min(gray)),
                "max_intensity": int(np.max(gray)),
                "dynamic_range": int(np.max(gray) - np.min(gray)),
                "median_intensity": float(np.median(gray))
            })
            
            # Histogram analysis
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            metrics["histogram_stats"] = {
                "entropy": float(-np.sum(hist * np.log2(hist + 1e-7))),
                "peak_count": len([i for i in range(1, 255) if hist[i] > hist[i-1] and hist[i] > hist[i+1]]),
                "uniformity": float(1.0 / (1.0 + np.var(hist)))
            }
            
            # Gradient analysis
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            metrics["gradient_stats"] = {
                "mean_gradient": float(np.mean(gradient_magnitude)),
                "std_gradient": float(np.std(gradient_magnitude)),
                "max_gradient": float(np.max(gradient_magnitude))
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Enhanced technical metrics calculation failed: {e}")
            return {"error": str(e)}

# Global enhanced media analyzer instance
_enhanced_media_analyzer = None

def get_enhanced_media_analyzer() -> EnhancedMediaAnalyzer:
    """Get or create global enhanced media analyzer instance."""
    global _enhanced_media_analyzer
    if _enhanced_media_analyzer is None:
        _enhanced_media_analyzer = EnhancedMediaAnalyzer()
    return _enhanced_media_analyzer