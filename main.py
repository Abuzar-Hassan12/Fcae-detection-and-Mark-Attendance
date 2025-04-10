import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import re
from threading import Lock
import os
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial

# ========================
# CONFIGURATION SETTINGS
# ========================
ENCODING_FILE = 'Encoding File.p'
IMAGES_FOLDER = 'images'
MAX_IMAGE_DIMENSION = 800
FACE_DETECTION_MODEL = 'hog'
NUM_ENCODING_JITTERS = 1
modeFolderPath = "Modes"
imagebackground = cv2.imread('Resources/background.png')
HOLIDAYS = ['23-03-2025', '01-05-2025', '14-08-2025']
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
MODE_DISPLAY_DURATION = 5
FACE_MATCH_THRESHOLD = 0.4
ATTENDANCE_START_TIME = datetime.strptime("6:00:00 AM", "%I:%M:%S %p").time()
ATTENDANCE_END_TIME = datetime.strptime("11:00:00 AM", "%I:%M:%S %p").time()
COOLDOWN_SECONDS = 30
EXCEL_LOCK = Lock()

# ========================
# GLOBAL STATE VARIABLES
# ========================
last_seen = {}
detected_students = set()
cooldown_tracker = {}
student_data = {}
current_mode = 0
mode_start_time = None
current_student_id = None
last_date_check = datetime.now()
founded_encodings = []
stud_ID = []

# ========================
# ENCODING GENERATION FUNCTIONS
# ========================
def process_single_image(images_folder, filename):
    try:
        filepath = os.path.join(images_folder, filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f"⚠️ Couldn't read: {filename}")
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        scale = MAX_IMAGE_DIMENSION / max(h, w)
        if scale < 1:
            img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        face_locations = face_recognition.face_locations(img_rgb, model=FACE_DETECTION_MODEL)
        if not face_locations:
            print(f"🚫 No face in {filename}")
            return None

        largest_face = max(face_locations, key=lambda loc: (loc[2]-loc[0])*(loc[1]-loc[3]))
        encoding = face_recognition.face_encodings(
            img_rgb, 
            known_face_locations=[largest_face],
            num_jitters=NUM_ENCODING_JITTERS
        )[0]
        return (encoding, os.path.splitext(filename)[0])
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None
def batch_process_images(images_folder, filenames):
    processor_count = min(cpu_count(), 8)
    process_task = partial(process_single_image, images_folder)
    with Pool(processes=processor_count) as pool:
        results = pool.map(process_task, filenames)
    successful_results = [r for r in results if r is not None]
    if not successful_results:
        return [], []
    encodings, ids = zip(*successful_results)
    return list(encodings), list(ids)  # Convert tuples to lists

def check_and_update_encodings():
    if not os.path.exists(ENCODING_FILE):
        print("Encoding file not found. Generating from all images...")
        image_files = os.listdir(IMAGES_FOLDER)
        encodings, ids = batch_process_images(IMAGES_FOLDER, image_files)
        data = {'encodings': encodings, 'ids': ids}
        with open(ENCODING_FILE, 'wb') as f:
            pickle.dump(data, f)
        print(f"Encodings saved to {ENCODING_FILE}")
    else:
        with open(ENCODING_FILE, 'rb') as f:
            data = pickle.load(f)
        # Ensure encodings and ids are lists (legacy files might have tuples)
        if isinstance(data['encodings'], tuple):
            data['encodings'] = list(data['encodings'])
        if isinstance(data['ids'], tuple):
            data['ids'] = list(data['ids'])
        
        existing_ids = set(data['ids'])
        current_images = os.listdir(IMAGES_FOLDER)
        current_ids = {os.path.splitext(img)[0] for img in current_images}
        new_images = [img for img in current_images if os.path.splitext(img)[0] not in existing_ids]
        
        if new_images:
            print(f"Found {len(new_images)} new images. Updating encoding file...")
            new_encodings, new_ids = batch_process_images(IMAGES_FOLDER, new_images)
            data['encodings'].extend(new_encodings)
            data['ids'].extend(new_ids)
            with open(ENCODING_FILE, 'wb') as f:
                pickle.dump(data, f)
            print(f"Updated encoding file with {len(new_ids)} new entries.")

# ========================
# INITIALIZATION FUNCTIONS
# ========================
def load_encodings():
    global founded_encodings, stud_ID
    with open(ENCODING_FILE, 'rb') as encode_file:
        encodings_dict = pickle.load(encode_file)
        founded_encodings, stud_ID = encodings_dict['encodings'], encodings_dict['ids']
    founded_encodings = np.array(founded_encodings, dtype=np.float64)

def load_mode_images():
    global imgModeList
    if not os.path.exists(modeFolderPath):
        raise FileNotFoundError(f"Directory not found: {modeFolderPath}")
    mode_files = sorted([f for f in os.listdir(modeFolderPath) if re.match(r"^mode[0-3]\.png$", f)], key=lambda x: int(x[4:-4]))
    if len(mode_files) != 4:
        raise ValueError(f"Need exactly 4 mode images. Found {len(mode_files)}")
    imgModeList = [cv2.imread(os.path.join(modeFolderPath, f)) for f in mode_files]
# def load_mode_images():
#     global imgModeList
#     if not os.path.exists(modeFolderPath):
#         raise FileNotFoundError(f"Directory not found: {modeFolderPath}")

#     mode_files = sorted([f for f in os.listdir(modeFolderPath) if re.match(r"^mode[0-3]\.png$", f)], 
#                        key=lambda x: int(x[4:-4]))
    
#     if len(mode_files) != 4:
#         raise ValueError(f"Need exactly 4 mode images. Found {len(mode_files)}")

#     imgModeList = [cv2.imread(os.path.join(modeFolderPath, f)) for f in mode_files]

# ========================
# ATTENDANCE CORE FUNCTIONS
# ========================
def clean_columns(df):
    df.rename(columns=lambda x: x.strip(), inplace=True)
    required_columns = ['Student ID', 'Name', 'Total Attendance']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing column: '{col}'")
    return df

def mark_off_if_needed(excel_file_path):
    today_date = datetime.now().strftime('%d-%m-%Y')
    weekday = datetime.now().weekday()
    
    if weekday >= 5 or today_date in HOLIDAYS:
        with EXCEL_LOCK:
            df = pd.read_excel(excel_file_path)
            df = clean_columns(df)

            status_col = f'{today_date} Status'
            time_col = f'{today_date} Time'

            if status_col not in df.columns:
                df[status_col] = 'Off'
                df[time_col] = ''
                df.to_excel(excel_file_path, index=False)
                print(f"Marked Off: {excel_file_path}")
                return True
        return False

def mark_absentees_before_attendance(excel_file_path):
    today_date = datetime.now().strftime('%d-%m-%Y')
    with EXCEL_LOCK:
        df = pd.read_excel(excel_file_path)
        df = clean_columns(df)

        status_col = f'{today_date} Status'
        time_col = f'{today_date} Time'

        if status_col not in df.columns:
            df[status_col] = 'A'
            df[time_col] = '00:00:00'
            updated = True
        else:
            updated = False

        if updated:
            df.to_excel(excel_file_path, index=False)
            print(f"Absentees marked: {excel_file_path}")

def preload_detected_students():
    global detected_students
    today_date = datetime.now().strftime('%d-%m-%Y')
    for excel_path in STEP_FILES.values():
        try:
            with EXCEL_LOCK:
                df = pd.read_excel(excel_path)
                df = clean_columns(df)
                status_col = f'{today_date} Status'
                if status_col in df.columns:
                    present_students = df[df[status_col] == 'P']['Student ID'].astype(str).tolist()
                    detected_students.update(present_students)
                    print(f"Preloaded detected students from {excel_path}: {present_students}")
        except Exception as e:
            print(f"Error preloading detected students from {excel_path}: {e}")

def is_already_present(student_id):
    """Return True if attendance for today is already marked as 'P'."""
    try:
        excel_path = get_excel_path(student_id)
        current_time = datetime.now()
        today_date = current_time.strftime('%d-%m-%Y')
        with EXCEL_LOCK:
            df = pd.read_excel(excel_path)
            df = clean_columns(df)
            status_col = f'{today_date} Status'
            if status_col in df.columns:
                student_row = df[df['Student ID'] == student_id]
                if not student_row.empty and student_row.iloc[0][status_col] == 'P':
                    return True
        return False
    except Exception as e:
        print(f"Error checking attendance for {student_id}: {e}")
        return False

def MarkAttendance(student_id):
    global detected_students
    with EXCEL_LOCK:
        current_time = datetime.now()
        if not (ATTENDANCE_START_TIME <= current_time.time() <= ATTENDANCE_END_TIME):
            print(f"Attendance closed for {student_id}")
            return

        excel_path = get_excel_path(student_id)
        today_date = current_time.strftime('%d-%m-%Y')
        
        try:
            df = pd.read_excel(excel_path)
            df = clean_columns(df)
            df['Student ID'] = df['Student ID'].astype(str)

            status_col = f'{today_date} Status'
            time_col = f'{today_date} Time'

            if status_col not in df.columns:
                df[status_col] = 'A'
                df[time_col] = '00:00:00'

            student_row = df[df['Student ID'] == student_id]
            if student_row.empty:
                print(f"Student {student_id} not found")
                return

            idx = student_row.index[0]
            if df.at[idx, status_col] == 'P':
                return

            df.at[idx, status_col] = 'P'
            df.at[idx, time_col] = current_time.strftime("%I:%M:%S %p")
            df.at[idx, 'Total Attendance'] += 1
            
            df.to_excel(excel_path, index=False)
            detected_students.add(student_id)
            print(f"Marked Present: {student_id}")

        except Exception as e:
            print(f"Excel Error: {str(e)}")

def LoadStudentName(student_id):
    try:
        excel_path = get_excel_path(student_id)
        with EXCEL_LOCK:
            df = pd.read_excel(excel_path)
            df = clean_columns(df)
            row = df[df['Student ID'] == student_id].iloc[0]
            return row['Name'], row['Total Attendance']
    except:
        return f"Student {student_id}", 0

# ========================
# FILE PATH CONFIGURATION
# ========================
STEP_FILES = {
    "FY10": "C:/Face-Detection-System/stud_data/FY1.xlsx",
    "FY20": "C:/Face-Detection-System/stud_data/FY2.xlsx", 
    "FY30": "C:/Face-Detection-System/stud_data/FY3.xlsx",
    "ST10": "C:/Face-Detection-System/stud_data/ST1.xlsx",
    "ST20": "C:/Face-Detection-System/stud_data/ST2.xlsx",
    "ST30": "C:/Face-Detection-System/stud_data/ST3.xlsx",
    "ST40": "C:/Face-Detection-System/stud_data/ST4.xlsx",
    "ST50": "C:/Face-Detection-System/stud_data/ST5.xlsx",
    "ST60": "C:/Face-Detection-System/stud_data/ST6.xlsx",
}

def get_excel_path(student_id):
    """Get full path to Excel file with proper validation"""
    for prefix, path in STEP_FILES.items():
        if student_id.startswith(prefix):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Excel file not found: {path}\n"
                    f"Required for student ID: {student_id}\n"
                    "Check:\n"
                    "1. File exists at specified path\n"
                    "2. Student ID prefixes match configuration"
                )
            return path
    raise ValueError(
        f"No Excel mapping for ID: {student_id}\n"
        "Valid prefixes:\n" +
        "\n".join([f"{k} -> {v}" for k, v in STEP_FILES.items()])
    )       

def get_major(student_id):
    return os.path.basename(get_excel_path(student_id)).split('.')[0]

# ========================
# VIDEO PROCESSING FUNCTIONS
# ========================
def reset_daily_state():
    global last_seen, detected_students, cooldown_tracker, last_date_check
    now = datetime.now()
    if now.date() != last_date_check.date():
        print("\n--- NEW DAY RESET ---")
        detected_students.clear()
        cooldown_tracker.clear()
        last_seen.clear()
        last_date_check = now
        
        step_files = [
            'C:/Face-Detection-System/stud_data/FY1.xlsx',
            'C:/Face-Detection-System/stud_data/FY2.xlsx',
            'C:/Face-Detection-System/stud_data/FY3.xlsx',
            'C:/Face-Detection-System/stud_data/ST1.xlsx',
            'C:/Face-Detection-System/stud_data/ST2.xlsx',
            'C:/Face-Detection-System/stud_data/ST3.xlsx',
            'C:/Face-Detection-System/stud_data/ST4.xlsx',
            'C:/Face-Detection-System/stud_data/ST5.xlsx',
            'C:/Face-Detection-System/stud_data/ST6.xlsx',
        ]
        for path in step_files:
            mark_absentees_before_attendance(path)

def process_frame(frame, frame_bg):
    global current_mode, mode_start_time, current_student_id

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(founded_encodings, encoding, FACE_MATCH_THRESHOLD)
        face_distances = face_recognition.face_distance(founded_encodings, encoding)

        if not matches[np.argmin(face_distances)]:
            continue
        
        student_id = stud_ID[np.argmin(face_distances)]
        last_seen[student_id] = time.time()

        top, right, bottom, left = [v * 4 for v in face_location]

        cv2.rectangle(frame_bg, (55 + left, 162 + top), (55 + right, 162 + bottom), (0, 255, 0), 2)
    
        cv2.putText(frame_bg, f"ID: {student_id}", (55 + left, 162 + top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if student_id not in detected_students:
            if is_already_present(student_id):
                if time.time() - cooldown_tracker.get(student_id, 0) > COOLDOWN_SECONDS:
                    current_mode = 3
                    mode_start_time = time.time()
                    cooldown_tracker[student_id] = time.time()
                detected_students.add(student_id)
            else:
                if current_student_id is None or current_student_id != student_id:
                    MarkAttendance(student_id)
                    detected_students.add(student_id)
                    current_mode = 1
                    mode_start_time = time.time()
                    current_student_id = student_id
                    
                    if student_id not in student_data:
                        name, attendance = LoadStudentName(student_id)
                        img_path = f'C:/Face-Detection-System/resized_images/{student_id}.jpg'
                        student_data[student_id] = {
                            'name': name,
                            'attendance': attendance,
                            'image': cv2.imread(img_path) if os.path.exists(img_path) else None
                        }
        else:
            if current_student_id == student_id and current_mode in [1, 2]:
                continue
            if time.time() - cooldown_tracker.get(student_id, 0) > COOLDOWN_SECONDS:
                current_mode = 3
                mode_start_time = time.time()
                cooldown_tracker[student_id] = time.time()

# ========================
# MAIN SYSTEM LOOP
# ========================
def main():
    global current_mode, mode_start_time, current_student_id
    check_and_update_encodings()
    load_encodings()
    load_mode_images()
    
    for path in STEP_FILES.values():
        if not os.path.exists(path):
            raise SystemExit(f"Critical error: Required file not found\nMissing: {path}")
    for path in STEP_FILES.values():
        if mark_off_if_needed(path):
            print("System exiting due to holiday/weekend")
            return
        mark_absentees_before_attendance(path)
    preload_detected_students()
    
    cap = cv2.VideoCapture(0)
    cap.set(3, FRAME_WIDTH)
    cap.set(4, FRAME_HEIGHT)
    time.sleep(2)
    
    while True:
        success, img = cap.read()
        if not success:
            continue
        reset_daily_state()
        frame_bg = imagebackground.copy()
        frame_bg[162:162+480, 55:55+640] = img
        process_frame(img, frame_bg)
        
        if current_mode in [1, 2, 3] and mode_start_time:
            elapsed = time.time() - mode_start_time
            if elapsed > MODE_DISPLAY_DURATION:
                if current_mode == 1:
                    current_mode = 2
                    mode_start_time = time.time()
                elif current_mode == 2:
                    current_mode = 0
                    if current_student_id:
                        cooldown_tracker[current_student_id] = time.time()
                    current_student_id = None
                    mode_start_time = None
                elif current_mode == 3:
                    current_mode = 0
                    mode_start_time = None
        frame_bg[44:44+633, 808:808+414] = imgModeList[current_mode]
        
        if current_mode == 1 and current_student_id:
            student = student_data.get(current_student_id)
            if student:
                if student['image'] is not None:
                    frame_bg[175:175+216, 909:909+216] = student['image']
                (w, h), _ = cv2.getTextSize(student['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                x_pos = 808 + (414 - w) // 2
                cv2.putText(frame_bg, str(student['attendance']), (861, 125), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
                cv2.putText(frame_bg, student['name'], (x_pos, 445), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,50), 1)
                cv2.putText(frame_bg, f"ID: {current_student_id}", (1006, 493), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(frame_bg, get_major(current_student_id), (1006, 550), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
        
        cv2.imshow("STEP School Talagang Campus", frame_bg)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        if cv2.getWindowProperty("STEP School Talagang Campus", cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()