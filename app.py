from flask import (
    Flask, render_template, Response, request,
    jsonify, session)
import cv2
import numpy as np
import time
import os
from PIL import Image
import io
import logging
import json
import face_recognition as fr
import atexit
from config import Config
from extensions import db, ma
from models import Person, Embedding
from schemas import person_schema, persons_schema
import face_utils


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


temp_unknown_face_storage_webcam = {"encoding": None, "timestamp": 0}
show_prompt_flag_webcam = False
video_capture_instance = None


def create_app(config_class=Config):
    flask_app = Flask(__name__, template_folder='templates')
    flask_app.config.from_object(config_class)
    flask_app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

    log.info("Initializing Flask extensions...")
    db.init_app(flask_app)
    if ma:
        ma.init_app(flask_app)

    with flask_app.app_context():
        log.info("Checking and creating database tables if necessary...")
        try:
            db.create_all()
            log.info("Database tables checked/created.")
            face_utils.update_known_faces_cache(force_update=True)
        except Exception as e:
            log.error(
                f"Database initialization or initial cache load failed: {e}", exc_info=True)

    def init_camera():
        global video_capture_instance
        log.debug("init_camera called.")
        if video_capture_instance is None or not video_capture_instance.isOpened():
            log.debug(
                "Attempting to initialize camera with cv2.VideoCapture(0, cv2.CAP_DSHOW)...")
            video_capture_instance = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not video_capture_instance.isOpened():
                log.error("Error: Could not open webcam.")
                if video_capture_instance is not None:
                    video_capture_instance.release()
                video_capture_instance = None
                return False
            log.debug("Camera initialized successfully.")
        else:
            log.debug("Camera was already initialized and opened.")
        return True

    def release_camera_on_exit():
        global video_capture_instance
        if video_capture_instance is not None and video_capture_instance.isOpened():
            log.info("Releasing camera resource on exit...")
            video_capture_instance.release()
            video_capture_instance = None
            cv2.destroyAllWindows()
            log.info("Camera released.")

    atexit.register(release_camera_on_exit)

    @flask_app.route('/')
    def index():
        log.debug("Route / called, serving index.html")
        return render_template('index.html')

    def generate_webcam_frames():
        global show_prompt_flag_webcam, temp_unknown_face_storage_webcam, video_capture_instance

        if not init_camera() or video_capture_instance is None:
            log.error("Camera not available for generating frames.")
            error_msg = "Error: Webcam not available."
            yield (b'--frame\r\nContent-Type: text/plain\r\n\r\n' + error_msg.encode() + b'\r\n')
            return

        log.info(
            "Webcam opened successfully by generate_webcam_frames. Starting stream.")
        last_prompt_time = 0
        prompt_cooldown_seconds = 5
        frame_processing_interval = 5
        current_frame_count = 0
        last_face_results_drawn = []

        window_name = "Webcam Feed (Press 's' to save snapshot, 'q' in window to stop stream)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        while True:
            if video_capture_instance is None or not video_capture_instance.isOpened():
                log.error("Video capture instance lost or closed unexpectedly.")
                break

            success, frame_bgr = video_capture_instance.read()
            if not success:
                log.warning("Failed to grab frame from webcam.")
                time.sleep(0.1)
                continue

            current_frame_count += 1
            display_frame = frame_bgr.copy()

            current_face_results_for_processing = []

            if current_frame_count % frame_processing_interval == 0:
                log.debug(f"Processing frame {current_frame_count}")
                try:
                    with flask_app.app_context():
                        current_face_results_for_processing = face_utils.process_frame_for_faces(
                            frame_bgr.copy())
                    last_face_results_drawn = current_face_results_for_processing
                except Exception as e:
                    log.error(
                        f"Error processing frame {current_frame_count}: {e}", exc_info=True)
                    current_face_results_for_processing = []
                    last_face_results_drawn = []

                unknown_face_encoding_this_frame = None
                for result in current_face_results_for_processing:
                    if result.get("name") == "Unknown" and result.get("encoding") is not None:
                        unknown_face_encoding_this_frame = result["encoding"]
                        log.debug(
                            f"Unknown face detected in frame {current_frame_count}. Distance to closest known: {result.get('distance', 'N/A')}")
                        break

                current_time = time.time()
                if unknown_face_encoding_this_frame is not None and \
                   (current_time - last_prompt_time > prompt_cooldown_seconds) and \
                   not show_prompt_flag_webcam:
                    log.info("Storing unknown face encoding for webcam prompt.")
                    temp_unknown_face_storage_webcam["encoding"] = unknown_face_encoding_this_frame
                    temp_unknown_face_storage_webcam["timestamp"] = current_time
                    show_prompt_flag_webcam = True
                    log.info("Set global prompt flag for webcam to TRUE.")
                    last_prompt_time = current_time

            display_frame = face_utils.draw_faces(
                display_frame, last_face_results_drawn)

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                snapshot_path = os.path.join(
                    os.getcwd(), "webcam_snapshot.jpg")
                cv2.imwrite(snapshot_path, frame_bgr)
                log.info(f"Snapshot saved to {snapshot_path}")
            elif key == ord('q'):
                log.info("Quit key 'q' pressed in OpenCV window. Stopping stream.")
                break

            ret, buffer = cv2.imencode('.jpg', display_frame)
            if not ret:
                log.warning("Failed to encode frame to JPEG.")
                continue

            frame_bytes = buffer.tobytes()
            try:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except GeneratorExit:
                log.info("Client disconnected from video stream.")
                break
            except Exception as e:
                log.error(f"Error yielding frame: {e}", exc_info=True)
                break

        log.info("Exiting generate_webcam_frames loop.")

    @flask_app.route('/video_feed')
    def video_feed():
        global temp_unknown_face_storage_webcam, show_prompt_flag_webcam
        log.debug("Route /video_feed called.")
        temp_unknown_face_storage_webcam = {"encoding": None, "timestamp": 0}
        show_prompt_flag_webcam = False
        return Response(generate_webcam_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @flask_app.route('/check_add_prompt')
    def check_add_prompt():
        global show_prompt_flag_webcam
        show = show_prompt_flag_webcam
        if show:
            log.debug("Responding to /check_add_prompt: Show=True")
        return jsonify({"show": show})

    @flask_app.route('/submit_webcam_prompt_face', methods=['POST'])
    def submit_webcam_prompt_face():
        global temp_unknown_face_storage_webcam, show_prompt_flag_webcam
        name = request.form.get('name', '').strip()

        if not name:
            return jsonify({"error": "Name is required."}), 400

        encoding_to_save = temp_unknown_face_storage_webcam.get("encoding")
        if encoding_to_save is None:
            return jsonify({"error": "Face data not found or prompt expired."}), 400

        if face_utils.save_face_to_db(name, encoding_to_save):
            log.info(f"Successfully added webcam face for user '{name}'.")
            temp_unknown_face_storage_webcam = {
                "encoding": None, "timestamp": 0}
            show_prompt_flag_webcam = False
            return jsonify({"message": f"Webcam face for '{name}' added!", "person": person_schema.dump(Person.query.filter_by(name=name).first())}), 201
        else:
            log.error(f"Failed to save webcam face for '{name}' to DB.")
            return jsonify({"error": "Failed to add face. Check server logs."}), 500

    @flask_app.route('/cancel_webcam_prompt', methods=['POST'])
    def cancel_webcam_prompt():
        global temp_unknown_face_storage_webcam, show_prompt_flag_webcam
        log.info("User cancelled adding face from webcam prompt.")
        temp_unknown_face_storage_webcam = {"encoding": None, "timestamp": 0}
        show_prompt_flag_webcam = False
        return jsonify({"message": "Add face action cancelled."})

    @flask_app.route('/upload', methods=['POST'])
    def upload_image():
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file:
            try:
                img_pil = Image.open(file.stream).convert('RGB')
                img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                log.info(
                    f"Processing uploaded image for recognition: {file.filename}")

                with flask_app.app_context():
                    face_results = face_utils.process_frame_for_faces(img_bgr)

                if not face_results:
                    return jsonify({"message": "No faces found in uploaded image."}), 200

                result = face_results[0]
                if result["name"] != "Unknown":
                    return jsonify({"message": f"Face recognized: {result['name']} (Distance: {result.get('distance', 'N/A'):.2f})"}), 200
                else:
                    return jsonify({
                        "message": "Unknown face detected in uploaded image.",
                        "ask_to_add": True,
                        "unknown_face_embedding": result["encoding"].tolist() if result["encoding"] is not None else None
                    }), 200
            except Exception as e:
                log.error(
                    f"Error processing single upload: {e}", exc_info=True)
                return jsonify({"error": f"Failed to process image: {str(e)}"}), 500
        return jsonify({"error": "File processing failed"}), 500

    @flask_app.route('/add_face_from_upload', methods=['POST'])
    def add_face_from_upload():
        name = request.form.get('name', '').strip()
        embedding_str = request.form.get('embedding')

        if not name:
            return jsonify({"error": "Name is required."}), 400
        if not embedding_str:
            return jsonify({"error": "Embedding data is required."}), 400

        try:
            embedding_list = json.loads(embedding_str)
            embedding_array = np.array(embedding_list, dtype=np.float64)

            if embedding_array.shape != (128,):
                log.warning(
                    f"Invalid embedding shape for {name} from upload: {embedding_array.shape}")
                return jsonify({"error": "Invalid embedding data format (shape)."}), 400

            if face_utils.save_face_to_db(name, embedding_array):
                person = Person.query.filter_by(name=name).first()
                return jsonify({"message": f"Face for '{name}' (from upload) added!", "person": person_schema.dump(person)}), 201
            else:
                return jsonify({"error": "Failed to save face from upload to DB."}), 500
        except json.JSONDecodeError:
            log.error(
                f"JSONDecodeError for embedding from upload for {name}.", exc_info=True)
            return jsonify({"error": "Invalid embedding data format (JSON)."}), 400
        except Exception as e:
            log.error(
                f"DB error adding face from upload for '{name}': {e}", exc_info=True)
            return jsonify({"error": f"Failed to add face from upload: {str(e)}"}), 500

    @flask_app.route('/cancel_add_face_from_upload', methods=['POST'])
    def cancel_add_face_from_upload():
        log.info("User cancelled adding face from upload.")
        return jsonify({"message": "Add face from upload action cancelled."})

    @flask_app.route('/batch_upload', methods=['POST'])
    def batch_upload():
        name = request.form.get('name', '').strip()
        files = request.files.getlist('files')
        if not name:
            return jsonify({"error": "Person's name is required."}), 400
        if not files or all(f.filename == '' for f in files):
            return jsonify({"error": "No files selected."}), 400

        processed_count = 0
        added_count = 0
        skipped_files = []

        try:
            for file_item in files:
                if file_item.filename == '':
                    continue
                processed_count += 1
                log.info(
                    f"Processing batch file: {file_item.filename} for {name}")
                try:
                    img_pil = Image.open(file_item.stream).convert('RGB')
                    img_bgr = cv2.cvtColor(
                        np.array(img_pil), cv2.COLOR_RGB2BGR)

                    rgb_frame_for_detection = cv2.cvtColor(
                        img_bgr, cv2.COLOR_BGR2RGB)
                    face_locations = fr.face_locations(
                        rgb_frame_for_detection, model="hog")

                    if not face_locations:
                        log.warning(
                            f"Skipping {file_item.filename}: No face detected.")
                        skipped_files.append(f"{file_item.filename} (no face)")
                        continue
                    if len(face_locations) > 1:
                        log.warning(
                            f"Skipping {file_item.filename}: Multiple faces ({len(face_locations)}). Expected 1.")
                        skipped_files.append(
                            f"{file_item.filename} (multiple faces)")
                        continue

                    top, right, bottom, left = face_locations[0]
                    face_image_crop_bgr = img_bgr[top:bottom, left:right]

                    if face_image_crop_bgr.size == 0:
                        log.warning(
                            f"Skipping {file_item.filename}: Face crop resulted in empty image.")
                        skipped_files.append(
                            f"{file_item.filename} (empty crop)")
                        continue

                    embedding_array = face_utils.get_embedding(
                        face_image_crop_bgr)

                    if embedding_array is not None and embedding_array.shape == (128,):
                        if face_utils.save_face_to_db(name, embedding_array):
                            added_count += 1
                        else:
                            skipped_files.append(
                                f"{file_item.filename} (DB save error)")
                    else:
                        log.warning(
                            f"Skipping {file_item.filename}: Encoding failed or invalid shape.")
                        skipped_files.append(
                            f"{file_item.filename} (encoding failed)")
                except Exception as file_e:
                    log.error(
                        f"Error processing file {file_item.filename}: {file_e}", exc_info=True)
                    skipped_files.append(
                        f"{file_item.filename} (processing error)")

            message = f"Processed {processed_count} files for '{name}'. Added {added_count} new face embeddings."
            if skipped_files:
                message += f" Skipped {len(skipped_files)} files: {', '.join(skipped_files)}"
            return jsonify({"message": message, "added": added_count, "skipped": len(skipped_files)}), 201
        except Exception as e:
            log.error(
                f"Error during batch upload for {name}: {e}", exc_info=True)
            return jsonify({"error": f"Failed batch upload: {str(e)}"}), 500

    @flask_app.route('/users', methods=['GET'])
    def list_users():
        try:
            all_persons = Person.query.all()
            return jsonify(persons_schema.dump(all_persons)), 200
        except Exception as e:
            log.error(f"Error fetching persons list: {e}", exc_info=True)
            return jsonify({"error": "Could not retrieve persons list"}), 500

    return flask_app


if __name__ == '__main__':
    app = create_app()
    with app.app_context():
        log.info("Initializing database and loading known faces cache...")
        db.create_all()
        face_utils.update_known_faces_cache(force_update=True)

    log.info("Starting Flask development server...")
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=4008)
