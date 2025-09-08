import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAAR_CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

def find_and_crop_card(image):
    """Finds a face and uses its position to crop the card area."""
    try:
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0: return None
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        img_h, img_w, _ = image.shape
        card_x1 = max(0, int(x - (w * 0.5)))
        card_y1 = max(0, int(y - (h * 0.8)))
        card_x2 = min(img_w, int(x + w + (w * 3.5)))
        card_y2 = min(img_h, int(y + h + (h * 1.5)))
        return image[card_y1:card_y2, card_x1:card_x2]
    except Exception as e:
        print(f"Error in find_and_crop_card: {e}")
        return None

def crop_main_photo(card_image):
    """Crops the main photograph from the already cropped card image."""
    try:
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        gray_card = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_card, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0: return None
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        padding = 10
        return card_image[y-padding:y+h+padding, x-padding:x+w+padding]
    except Exception as e:
        print(f"Error in crop_main_photo: {e}")
        return None

def perform_ela(photo_image):
    """Performs ELA and returns a colored heatmap and a tamper score."""
    try:
        if photo_image is None or photo_image.size == 0:
            return None, -1

        pil_photo = Image.fromarray(cv2.cvtColor(photo_image, cv2.COLOR_BGR2RGB)).convert('RGB')
        
        temp_file_path = "temp_ela.jpg"
        pil_photo.save(temp_file_path, 'JPEG', quality=95)
        resaved_image = Image.open(temp_file_path).convert('RGB')
        os.remove(temp_file_path)

        # Calculate the raw difference
        ela_image_raw = ImageChops.difference(pil_photo, resaved_image)
        
        # --- ACCURATE SCORE CALCULATION ---
        # Enhance moderately for score calculation to avoid picking up natural edges
        score_enhancer = ImageEnhance.Brightness(ela_image_raw)
        ela_for_score = score_enhancer.enhance(15) # Gentle enhancement for score
        gray_ela_for_score = cv2.cvtColor(np.array(ela_for_score), cv2.COLOR_RGB2GRAY)
        suspicious_pixels = np.sum(gray_ela_for_score > 40) # Use a slightly higher brightness threshold
        score = (suspicious_pixels / gray_ela_for_score.size) * 100

        # --- VISUAL HEATMAP GENERATION ---
        # Enhance strongly for a clear visual heatmap
        display_enhancer = ImageEnhance.Brightness(ela_image_raw)
        ela_for_display = display_enhancer.enhance(50) # Strong enhancement for display
        ela_cv_display = cv2.cvtColor(np.array(ela_for_display), cv2.COLOR_RGB2BGR)
        gray_ela_display = cv2.cvtColor(ela_cv_display, cv2.COLOR_BGR2GRAY)
        
        ela_normalized = cv2.normalize(gray_ela_display, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(ela_normalized, cv2.COLORMAP_HOT)
        
        original_resized = cv2.resize(photo_image, (heatmap.shape[1], heatmap.shape[0]))
        combined_image = cv2.addWeighted(original_resized, 0.6, heatmap, 0.4, 0)
        
        final_ela_pil = Image.fromarray(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        
        return final_ela_pil, score
    except Exception as e:
        print(f"Error during ELA: {e}")
        return None, -1

def run_full_analysis(image_path):
    """Main orchestrator function for the analysis pipeline."""
    original_image = cv2.imread(image_path)
    if original_image is None:
        return {'error': 'Could not read the uploaded image file.'}
        
    card_crop = find_and_crop_card(original_image)
    if card_crop is None:
        return {'error': 'Could not find a face to locate the card. Please use a clearer photo.'}
        
    main_photo = crop_main_photo(card_crop)
    if main_photo is None:
        return {'error': 'Could not isolate the face from the detected card area.'}
        
    ela_image, ela_score = perform_ela(main_photo)
    if ela_image is None:
        return {'error': 'Failed to perform ELA analysis on the detected face.'}
        
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    main_photo_path = os.path.join(static_dir, 'main_photo.png')
    ela_image_path = os.path.join(static_dir, 'ela_result.png')
    cv2.imwrite(main_photo_path, main_photo)
    ela_image.save(ela_image_path)
    
    # Calibrate this threshold by testing on several known-good images
    tamper_threshold = 5.0 
    verdict = "Likely Authentic"
    if ela_score > tamper_threshold:
        verdict = "Suspicious / Likely Forged"
        
    return {
        'verdict': verdict,
        'ela_score': f"{ela_score:.2f}%",
        'main_photo_path': 'main_photo.png',
        'ela_image_path': 'ela_result.png'
    }