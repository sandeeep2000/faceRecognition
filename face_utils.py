# import face_recognition
# import numpy as np
# import json
# import cv2 
# import time
# import logging

# from flask import current_app 
# from config import Config
# from extensions import db
# from models import Person, Embedding

# log = logging.getLogger(__name__)


# known_face_data_cache = {
#     "names": [],
#     "encodings": [],
#     "last_update": 0
# }
# CACHE_TTL_SECONDS = 60
# is_data_loaded = False

# def load_known_faces_from_db():
   
#     global known_face_data_cache, is_data_loaded

#     log.info("Loading known faces (face_recognition) from DB into cache...")
#     try:
#         results = db.session.query(Person.name, Embedding.embedding_data)\
#                             .join(Embedding, Person.id == Embedding.person_id)\
#                             .all()
#         current_names = []
#         current_encodings = []

#         for name, embedding_json_str in results:
#             try:
#                 embedding_data_np = np.array(json.loads(embedding_json_str), dtype=np.float64)
#                 if embedding_data_np.shape == (128,):
#                     current_names.append(name)
#                     current_encodings.append(embedding_data_np)
#                 else:
#                     log.warning(f"Person '{name}' has embedding with incorrect shape {embedding_data_np.shape}. Skipping.")
#             except (json.JSONDecodeError, TypeError) as e:
#                 log.warning(f"Person '{name}' has invalid embedding JSON string '{embedding_json_str}': {e}. Skipping.")
        
#         known_face_data_cache["names"] = current_names
#         known_face_data_cache["encodings"] = current_encodings
#         known_face_data_cache["last_update"] = time.time()
#         log.info(f"Cache updated with {len(known_face_data_cache['encodings'])} face_recognition encodings.")
#         is_data_loaded = True
#     except Exception as e:
#         log.error(f"Error loading faces from DB for cache: {e}", exc_info=True)
#         is_data_loaded = False

# def update_known_faces_cache(force_update=False):
    
#     global known_face_data_cache, is_data_loaded
#     current_time = time.time()

#     if force_update or not is_data_loaded or \
#        (current_time - known_face_data_cache.get("last_update", 0)) >= CACHE_TTL_SECONDS:
#         load_known_faces_from_db()

# def save_face_to_db(name, encoding_array):
    
#     try:
#         person = Person.query.filter_by(name=name).first()
#         if not person:
#             log.info(f"Creating new person in DB: {name}")
#             person = Person(name=name)
#             db.session.add(person)
#             db.session.flush() 
        
#         if not isinstance(encoding_array, np.ndarray):
#             encoding_array = np.array(encoding_array, dtype=np.float64)

#         if encoding_array.shape != (128,):
#             log.error(f"Attempted to save encoding with incorrect shape {encoding_array.shape} for '{name}'. Expected (128,).")
#             return False

#         new_embedding_db_entry = Embedding(person_id=person.id, embedding_data=encoding_array)
#         db.session.add(new_embedding_db_entry)
#         db.session.commit()
        
#         log.info(f"Saved encoding for {name} to Database.")
#         update_known_faces_cache(force_update=True) 
#         return True
#     except Exception as e:
#         db.session.rollback()
#         log.error(f"Error saving face to Database for '{name}': {e}", exc_info=True)
#         return False

# def get_embedding(face_img_np_bgr):
    
#     try:
#         if face_img_np_bgr is None or face_img_np_bgr.size == 0:
#             log.error("get_embedding received an empty or None image.")
#             return None

#         if face_img_np_bgr.ndim == 3 and face_img_np_bgr.shape[2] == 3:
#             rgb_face = cv2.cvtColor(face_img_np_bgr, cv2.COLOR_BGR2RGB)
#         elif face_img_np_bgr.ndim == 2:
#             rgb_face = cv2.cvtColor(face_img_np_bgr, cv2.COLOR_GRAY2RGB)
#         else:
#             log.error(f"Invalid image dimensions for encoding: {face_img_np_bgr.shape}.")
#             return None

#         h, w, channels = rgb_face.shape
#         if h == 0 or w == 0:
#             log.error("Invalid face crop (height or width is 0) passed to get_embedding.")
#             return None
            
#         face_location_for_encoding = [(0, w, h, 0)] 
#         encodings = face_recognition.face_encodings(rgb_face, known_face_locations=face_location_for_encoding, num_jitters=1)

#         if encodings:
#             return encodings[0]
#         else:
#             log.warning("face_recognition.face_encodings failed for the provided face crop.")
#             return None
#     except Exception as e:
#         log.error(f"Error in get_embedding: {e}", exc_info=True)
#         return None

# def process_frame_for_faces(frame_bgr):
    
#     update_known_faces_cache() 
#     cached_encodings = known_face_data_cache["encodings"]
#     cached_names = known_face_data_cache["names"]

#     resize_factor = 0.4 
#     try:
#         small_frame = cv2.resize(frame_bgr, (0, 0), fx=resize_factor, fy=resize_factor)
#     except cv2.error as resize_err:
#         log.error(f"OpenCV error resizing frame: {resize_err}")
#         small_frame = frame_bgr 
#         resize_factor = 1.0
#     except Exception as resize_err:
#         log.error(f"Generic error resizing frame: {resize_err}")
#         small_frame = frame_bgr 
#         resize_factor = 1.0

#     rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
#     face_locations_small = face_recognition.face_locations(rgb_small_frame, model="hog") 
#     current_face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations_small)

#     face_results = []

#     for i, face_encoding_live in enumerate(current_face_encodings):
#         name = "Unknown"
#         min_distance_for_this_face = float('inf') 

#         if cached_encodings:
#             matches = face_recognition.compare_faces(
#                 cached_encodings,
#                 face_encoding_live,
#                 tolerance=current_app.config.get('FACE_RECOGNITION_TOLERANCE', Config.FACE_RECOGNITION_TOLERANCE)
#             )
#             face_distances_to_known = face_recognition.face_distance(cached_encodings, face_encoding_live)
            
#             if True in matches:
#                 best_match_index = np.argmin(face_distances_to_known)
#                 if matches[best_match_index]: 
#                     name = cached_names[best_match_index]
#                     min_distance_for_this_face = face_distances_to_known[best_match_index]
#             else: 
#                 if len(face_distances_to_known) > 0:
#                     min_distance_for_this_face = np.min(face_distances_to_known) 
        
#         top_s, right_s, bottom_s, left_s = face_locations_small[i]
#         facial_area = {
#             'x': int(left_s / resize_factor),
#             'y': int(top_s / resize_factor),
#             'w': int((right_s - left_s) / resize_factor),
#             'h': int((bottom_s - top_s) / resize_factor)
#         }

#         face_results.append({
#             "name": name,
#             "facial_area": facial_area,
#             "distance": min_distance_for_this_face if name != "Unknown" else (min_distance_for_this_face if min_distance_for_this_face != float('inf') else None),
#             "encoding": face_encoding_live if name == "Unknown" else None,
#             "confidence": 1.0 
#         })
#     return face_results

# def draw_faces(frame, face_results):
#     for result in face_results:
#         try:
#             area = result["facial_area"]
#             x, y, w, h = area['x'], area['y'], area['w'], area['h']
#             name = result["name"]
#             distance = result.get("distance")

#             box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

#             label = name
#             if name != "Unknown" and distance is not None:
#                 label += f" ({distance:.2f})"
            
#             font_face = cv2.FONT_HERSHEY_SIMPLEX
#             font_scale = 0.6
#             thickness = 1
#             (label_width, label_height), baseline = cv2.getTextSize(label, font_face, font_scale, thickness)
            
#             label_y_start = y - baseline - 5 
#             if label_y_start < label_height : 
#                 label_y_start = y + h + baseline + label_height + 5

#             label_bg_y_top = label_y_start - label_height - baseline
#             label_bg_y_bottom = label_y_start + baseline // 2

#             cv2.rectangle(frame, (x, label_bg_y_top),
#                           (x + label_width, label_bg_y_bottom), box_color, cv2.FILLED)
#             cv2.putText(frame, label, (x, label_y_start - baseline//2),
#                         font_face, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

#         except Exception as e:
#             log.error(f"Error drawing face result ({result.get('name', 'N/A')}): {e}", exc_info=True)
#     return frame


import face_recognition 
import numpy as np
import os
import json
from flask import current_app
import cv2 
import time
import logging
from config import Config 
from extensions import db
from models import Person, Embedding 

log = logging.getLogger(__name__)

known_face_data_cache = {
    "names": [],        
    "encodings": [],    
    "last_update": 0
}
CACHE_TTL_SECONDS = 60 
is_data_loaded = False 

def load_known_faces_from_db():
    
    global known_face_data_cache, is_data_loaded
    from flask import current_app 

    with current_app.app_context(): 
        log.info("Loading known faces (face_recognition) from DB into cache...")
        try:
            results = db.session.query(Person.name, Embedding._embedding_data)\
                                .join(Embedding, Person.id == Embedding.person_id)\
                                .all()

            current_names = []
            current_encodings = []

            for name, embedding_json_str in results:
                try:
                    embedding_data_np = np.array(json.loads(embedding_json_str), dtype=np.float64)
                    if embedding_data_np.shape == (128,):
                        current_names.append(name)
                        current_encodings.append(embedding_data_np)
                    else:
                        log.warning(f"Person '{name}' has embedding with incorrect shape {embedding_data_np.shape}. Skipping.")
                except (json.JSONDecodeError, TypeError) as e:
                    log.warning(f"Person '{name}' has invalid embedding JSON string '{embedding_json_str}': {e}. Skipping.")
            
            known_face_data_cache["names"] = current_names
            known_face_data_cache["encodings"] = current_encodings
            known_face_data_cache["last_update"] = time.time()
            log.info(f"Cache updated with {len(known_face_data_cache['encodings'])} face_recognition encodings.")
            is_data_loaded = True
        except Exception as e:
            log.error(f"Error loading faces from DB for cache: {e}", exc_info=True)
            is_data_loaded = False

def update_known_faces_cache(force_update=False):
    """Loads or refreshes known faces from Person/Embedding tables into cache."""
    global known_face_data_cache, is_data_loaded 
    current_time = time.time()

    if force_update or not is_data_loaded or \
       (current_time - known_face_data_cache.get("last_update", 0)) >= CACHE_TTL_SECONDS:
        load_known_faces_from_db()

def save_face_to_db(name, encoding_array):
    

    with current_app.app_context(): 
        try:
            person = Person.query.filter_by(name=name).first()
            if not person:
                log.info(f"Creating new person in DB: {name}")
                person = Person(name=name)
                db.session.add(person)
                db.session.flush() 
            
            if not isinstance(encoding_array, np.ndarray):
                encoding_array = np.array(encoding_array, dtype=np.float64)

            if encoding_array.shape != (128,):
                log.error(f"Attempted to save encoding with incorrect shape {encoding_array.shape} for {name}")
                return False

            new_embedding_db_entry = Embedding(person_id=person.id, embedding_data=encoding_array)
            db.session.add(new_embedding_db_entry)
            db.session.commit()
            
            log.info(f"Saved encoding for {name} to Database.")
            update_known_faces_cache(force_update=True) 
            return True
        except Exception as e:
            db.session.rollback()
            log.error(f"Error saving face to Database for {name}: {e}", exc_info=True)
            return False

def get_embedding(face_img_np_bgr): 
    
    try:
        if face_img_np_bgr is None or face_img_np_bgr.size == 0:
            log.error("get_embedding received an empty or None image.")
            return None

        if face_img_np_bgr.ndim == 3 and face_img_np_bgr.shape[2] == 3: 
             rgb_face = cv2.cvtColor(face_img_np_bgr, cv2.COLOR_BGR2RGB)
        elif face_img_np_bgr.ndim == 2: 
             rgb_face = cv2.cvtColor(face_img_np_bgr, cv2.COLOR_GRAY2RGB)
        else:
             log.error(f"Invalid image dimensions for encoding: {face_img_np_bgr.shape}")
             return None

        h, w, channels = rgb_face.shape
        if h == 0 or w == 0:
            log.error("Invalid face crop (height or width is 0) passed to get_embedding.")
            return None
            
        
        face_location_for_encoding = [(0, w, h, 0)] 
        encodings = face_recognition.face_encodings(rgb_face, known_face_locations=face_location_for_encoding, num_jitters=1)

        if encodings:
            return encodings[0] 
        else:
            log.warning("face_recognition.face_encodings failed for the provided face crop.")
            return None
    except Exception as e:
        log.error(f"Error in get_embedding: {e}", exc_info=True)
        return None

def process_frame_for_faces(frame_bgr):
   
    update_known_faces_cache() 
    cached_encodings = known_face_data_cache["encodings"]
    cached_names = known_face_data_cache["names"]

    resize_factor = 0.4 
    try:
        small_frame = cv2.resize(frame_bgr, (0, 0), fx=resize_factor, fy=resize_factor)
    except cv2.error as resize_err: 
        log.error(f"OpenCV error resizing frame: {resize_err}")
        small_frame = frame_bgr 
        resize_factor = 1.0
    except Exception as resize_err: 
        log.error(f"Generic error resizing frame: {resize_err}")
        small_frame = frame_bgr 
        resize_factor = 1.0

    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations_small = face_recognition.face_locations(rgb_small_frame, model="hog") 
    current_face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations_small)

    face_results = []

    for i, face_encoding_live in enumerate(current_face_encodings): 
        name = "Unknown"
        min_distance_for_this_face = float('inf') 

        if cached_encodings:
            matches = face_recognition.compare_faces(
                cached_encodings,
                face_encoding_live,
                tolerance=Config.FACE_RECOGNITION_TOLERANCE 
            )
            face_distances_to_known = face_recognition.face_distance(cached_encodings, face_encoding_live)
            
            if True in matches: 
                best_match_index = np.argmin(face_distances_to_known) 
                if matches[best_match_index]: 
                    name = cached_names[best_match_index]
                    min_distance_for_this_face = face_distances_to_known[best_match_index]
                    log.debug(f"Match: {name} (Dist: {min_distance_for_this_face:.4f})")
            else: 
                if len(face_distances_to_known) > 0:
                    min_distance_for_this_face = np.min(face_distances_to_known) 
                    log.debug(f"Unknown. Closest known was at dist: {min_distance_for_this_face:.4f}")
        else:
            log.debug("No known faces in cache to compare against.")

        top_s, right_s, bottom_s, left_s = face_locations_small[i]
        facial_area = {
            'x': int(left_s / resize_factor),
            'y': int(top_s / resize_factor),
            'w': int((right_s - left_s) / resize_factor),
            'h': int((bottom_s - top_s) / resize_factor)
        }

        face_results.append({
            "name": name,
            "facial_area": facial_area, 
            "distance": min_distance_for_this_face if name != "Unknown" else None, 
            "encoding": face_encoding_live if name == "Unknown" else None, 
            "confidence": 1.0 
        })
    return face_results

def draw_faces(frame, face_results):

    for result in face_results:
        try:
            area = result["facial_area"]
            x, y, w, h = area['x'], area['y'], area['w'], area['h']
            name = result["name"]
            distance = result.get("distance") 

            box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            label = name
            if name != "Unknown" and distance is not None:
                 label += f" ({distance:.2f})" 

            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (label_width, label_height), baseline = cv2.getTextSize(label, font_face, font_scale, thickness)
            
            
            label_y_start = y + h + baseline + thickness + 5 
            if label_y_start + label_height > frame.shape[0]: 
                label_y_start = y - baseline - 5 

            label_bg_y_top = label_y_start - label_height - baseline
            label_bg_y_bottom = label_y_start + baseline // 2

            cv2.rectangle(frame, (x, label_bg_y_top), 
                          (x + label_width, label_bg_y_bottom), box_color, cv2.FILLED)
            cv2.putText(frame, label, (x, label_y_start - baseline//2), 
                        font_face, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        except Exception as e:
            log.error(f"Error drawing face result {result}: {e}", exc_info=True)
    return frame
