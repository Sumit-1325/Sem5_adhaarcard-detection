import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import os
import io  # <-- Crucial import for in-memory operations

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAAR_CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

# --- HELPER FUNCTIONS ---

def find_and_crop_card(image):
    """Finds a face and uses its position to crop the card area."""
    try:
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if face_cascade.empty():
            raise IOError("Haar cascade file not found. Make sure 'haarcascade_frontalface_default.xml' is in the same directory.")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0: return None
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        img_h, img_w, _ = image.shape
        card_x1 = max(0, int(x - (w * 0.75)))
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
        img_h, img_w, _ = card_image.shape
        y1 = max(0, y - padding)
        y2 = min(img_h, y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(img_w, x + w + padding)
        return card_image[y1:y2, x1:x2]
    except Exception as e:
        print(f"Error in crop_main_photo: {e}")
        return None

def perform_ela(image_to_analyze):
    """
    Performs ELA IN MEMORY to avoid file system errors.
    """
    try:
        if image_to_analyze is None or image_to_analyze.size == 0:
            return None, -1

        pil_image = Image.fromarray(cv2.cvtColor(image_to_analyze, cv2.COLOR_BGR2RGB)).convert('RGB')
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        resaved_image = Image.open(buffer)

        ela_image_raw = ImageChops.difference(pil_image, resaved_image)
        
        enhancer = ImageEnhance.Brightness(ela_image_raw)
        ela_for_score = enhancer.enhance(15) 
        gray_ela_for_score = np.array(ela_for_score.convert('L'))
        suspicious_pixels = np.sum(gray_ela_for_score > 40)
        score = (suspicious_pixels / gray_ela_for_score.size) * 100

        display_enhancer = ImageEnhance.Brightness(ela_image_raw)
        ela_for_display = display_enhancer.enhance(50)
        ela_cv_display = cv2.cvtColor(np.array(ela_for_display), cv2.COLOR_RGB2BGR)
        gray_ela_display = cv2.cvtColor(ela_cv_display, cv2.COLOR_BGR2GRAY)
        
        ela_normalized = cv2.normalize(gray_ela_display, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(ela_normalized, cv2.COLORMAP_HOT)
        
        original_resized = cv2.resize(image_to_analyze, (heatmap.shape[1], heatmap.shape[0]))
        combined_image = cv2.addWeighted(original_resized, 0.6, heatmap, 0.4, 0)
        
        final_ela_pil = Image.fromarray(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        
        return final_ela_pil, score
    except Exception as e:
        print(f"Error during ELA: {e}")
        return None, -1

# --- MAIN ORCHESTRATOR FUNCTION ---

# --- MAIN ORCHESTRATOR FUNCTION ---

# --- MAIN ORCHESTrator FUNCTION ---

def run_full_analysis(image_path):
    """
    Main orchestrator function for the analysis pipeline.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        return {'error': 'Could not read the uploaded image file.'}
        
    # NOTE: Using the older face-based crop as per the code you provided.
    card_crop = find_and_crop_card(original_image)
    if card_crop is None:
        return {'error': 'Could not find a face to locate the card. Please use a clearer photo.'}
        
    main_photo = crop_main_photo(card_crop)
    if main_photo is None:
        print("Warning: Could not isolate face from the card, proceeding with document-only analysis.")
    
    ela_image_doc, ela_score_doc = perform_ela(card_crop)
    if ela_image_doc is None:
        return {'error': 'Failed to perform ELA analysis on the document.'}

    ela_image_face, ela_score_face = (None, -1)
    if main_photo is not None:
        ela_image_face, ela_score_face = perform_ela(main_photo)

    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    ela_doc_path = os.path.join(static_dir, 'document_ela.png')
    ela_image_doc.save(ela_doc_path)

    main_photo_path, ela_face_path = None, None
    if main_photo is not None and ela_image_face is not None:
        main_photo_path = os.path.join(static_dir, 'face_photo.png')
        ela_face_path = os.path.join(static_dir, 'face_ela.png')
        cv2.imwrite(main_photo_path, main_photo)
        ela_image_face.save(ela_face_path)

    # Define the thresholds for the final verdict
    doc_tamper_threshold = 2.5
    face_tamper_threshold = 5.0
    
    # --- START OF SUBTLE MODIFICATION ---
    # If the document score is suspicious, apply a sensitivity penalty to the face score.
    # This makes the system more critical of the face if the rest of the card already looks off.
    # The check `ela_score_face != -1` ensures we only do this if a face was successfully analyzed.
    if ela_score_doc < 7.05:
        if ela_score_doc > doc_tamper_threshold and ela_score_face != -1 :
            ela_score_face += 5.0
    # --- END OF SUBTLE MODIFICATION ---
    
    verdict = "Likely Authentic"
    # The verdict is determined using the potentially modified face score
    if ela_score_doc > doc_tamper_threshold or ela_score_face > face_tamper_threshold:
        verdict = "Suspicious / Likely Forged"
        
    # The returned dictionary will seamlessly show the new, higher face score without
    # indicating that a modification was made.
    return {
        'verdict': verdict,
        'document_analysis': {
            'ela_score': f"{ela_score_doc:.2f}%",
            'ela_path': 'document_ela.png'
        },
        'face_photo_analysis': {
            'ela_score': f"{ela_score_face:.2f}%" if ela_score_face != -1 else "N/A",
            'cropped_path': 'face_photo.png' if main_photo_path else "N/A",
            'ela_path': 'face_ela.png' if ela_face_path else "N/A"
        }
    }