import cv2
#import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import re
from threading import Lock
from insightface.app import FaceAnalysis

# ========================
# CONFIGURATION SETTINGS
# ========================
ENCODING_FILE = 'Encoding File.p'
modeFolderPath = "C:\\Face-Detection-System\\Resources\\Modes"
imagebackground = cv2.imread('Resources/background.png')  
HOLIDAYS = ['23-03-2025', '01-05-2025', '14-08-2025']
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
MODE_DISPLAY_DURATION = 5
FACE_MATCH_THRESHOLD = 0.6  # Cosine similarity threshold
ATTENDANCE_START_TIME = datetime.strptime("6:00:00 AM", "%I:%M:%S %p").time()
ATTENDANCE_END_TIME = datetime.strptime("11:00:00 AM", "%I:%M:%S %p").time()
COOLDOWN_SECONDS = 30  # Increased cooldown period
EXCEL_LOCK = Lock()  # Thread safety for Excel operations

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
face_analyzer = None

# ========================
# INITIALIZATION FUNCTIONS
# ========================
def load_encodings():
    global founded_encodings, stud_ID
    with open(ENCODING_FILE, 'rb') as encode_file:
        encodings_dict = pickle.load(encode_file)
        founded_encodings, stud_ID = encodings_dict['encodings'], encodings_dict['ids']
    
    # Convert to NumPy float32 array for InsightFace compatibility
    founded_encodings = np.array(founded_encodings, dtype=np.float32)

def load_mode_images():
    global imgModeList
    if not os.path.exists(modeFolderPath):
        raise FileNotFoundError(f"Directory not found: {modeFolderPath}")

    mode_files = sorted([f for f in os.listdir(modeFolderPath) if re.match(r"^mode[0-3]\.png$", f)], 
                       key=lambda x: int(x[4:-4]))
    
    if len(mode_files) != 4:
        raise ValueError(f"Need exactly 4 mode images. Found {len(mode_files)}")

    imgModeList = [cv2.imread(os.path.join(modeFolderPath, f)) for f in mode_files]

def init_face_analyzer():
    global face_analyzer
    face_analyzer = FaceAnalysis(name="buffalo_l")
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

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
    
    # Detect faces using InsightFace
    faces = face_analyzer.get(small_frame)
    
    for face in faces:
        current_embedding = face.embedding
        
        if founded_encodings.size == 0:
            continue  # No encodings loaded
        
        # Normalize embeddings for cosine similarity calculation
        current_embedding_norm = current_embedding / np.linalg.norm(current_embedding)
        stored_encodings_norm = founded_encodings / np.linalg.norm(founded_encodings, axis=1, keepdims=True)
        
        # Calculate similarities
        similarities = np.dot(stored_encodings_norm, current_embedding_norm)
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity < FACE_MATCH_THRESHOLD:
            continue
        
        student_id = stud_ID[best_match_idx]
        last_seen[student_id] = time.time()

        # Scale face coordinates to original frame size
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        x1 *= 4
        y1 *= 4
        x2 *= 4
        y2 *= 4

        # Draw rectangle on frame
        cv2.rectangle(frame_bg, (55 + x1, 162 + y1), (55 + x2, 162 + y2), (0, 255, 0), 2)
        cv2.putText(frame_bg, f"ID: {student_id}", (55 + x1, 162 + y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Attendance logic remains unchanged
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

    load_encodings()
    load_mode_images()
    init_face_analyzer()  # Initialize InsightFace model
    
    for path in STEP_FILES.values():
        if not os.path.exists(path):
            raise SystemExit(
                f"Critical error: Required file not found\n"
                f"Missing: {path}\n"
                "Please check:\n"
                "1. File exists at specified location\n"
                "2. Path matches your system configuration\n"
                "3. No typos in file names"
            )

    for path in STEP_FILES.values():
        if mark_off_if_needed(path):
            print("System exiting due to holiday/weekend")
            return
        mark_absentees_before_attendance(path)
    
    # Preload students already marked 'P' for today
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

        # Mode timing logic remains unchanged
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
        
        # Student info display logic remains unchanged
        if current_mode == 1 and current_student_id:
            student = student_data.get(current_student_id)
            if student:
                if student['image'] is not None:
                    frame_bg[175:175+216, 909:909+216] = student['image']
                
                (w, h), _ = cv2.getTextSize(student['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                x_pos = 808 + (414 - w) // 2
                
                cv2.putText(frame_bg, str(student['attendance']), (861, 125),
                           cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
                cv2.putText(frame_bg, student['name'], (x_pos, 445),
                           cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,50), 1)
                cv2.putText(frame_bg, f"ID: {current_student_id}", (1006, 493),
                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(frame_bg, get_major(current_student_id), (1006, 550),
                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("STEP School Talagang Campus", frame_bg)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            print("User requested exit. Exiting...")
            break
        if cv2.getWindowProperty("STEP School Talagang Campus", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed by user. Exiting...")
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()