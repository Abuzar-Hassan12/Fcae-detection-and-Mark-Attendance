import cv2
import os
import pickle
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import re

# =============================
# CONFIGURATION SETTINGS
# =============================
ENCODING_FILE = 'Encoding_File.p'
modeFolderPath = "C:\\Face-Detection-System\\Resources\\Modes"
imagebackground = cv2.imread('Resources/background.png')  
HOLIDAYS = ['23-03-2025', '01-05-2025', '14-08-2025']
COOLDOWN_MINUTES = 5
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
MODE_DISPLAY_DURATION = 3  # Mode 1 should display for 3 seconds
FACE_MATCH_THRESHOLD = 0.4  
ATTENDANCE_START_TIME = datetime.strptime("6:00:00 AM", "%I:%M:%S %p").time()
ATTENDANCE_END_TIME = datetime.strptime("8:30:00 AM", "%I:%M:%S %p").time()

# Dictionary to track last seen time of each student
last_seen = {}
# Initialize Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not access the camera.")
    exit()
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)
time.sleep(2)

# Load Encodings
with open(ENCODING_FILE, 'rb') as encode_file:
    founded_encodings, stud_ID = pickle.load(encode_file)

# ========================
# LOAD UI RESOURCES
# ========================
if not os.path.exists(modeFolderPath):
    raise FileNotFoundError(f"Directory not found: {modeFolderPath}")

all_files = os.listdir(modeFolderPath)
mode_files = [f for f in all_files if re.match(r"^mode[0-3]\.png$", f)]
mode_files.sort(key=lambda x: int(x[4:-4]))

if len(mode_files) != 4:
    raise ValueError(f"Need exactly 4 mode images (0-3). Found {len(mode_files)}: {mode_files}")

imgModeList = [cv2.imread(os.path.join(modeFolderPath, path)) for path in mode_files]

# ========================
# TRACKERS & CACHES
# ========================
detected_students = set()
cooldown_tracker = {}
student_data = {}
current_mode = 0
mode_timeout = 0
COOLDOWN_SECONDS = 5  # Cooldown period in seconds

# =================================
# CLEAN COLUMN NAMES IN EXCEL FILE
# =================================
def clean_columns(df):
    df.rename(columns=lambda x: x.strip(), inplace=True)
    required_columns = ['Student ID', 'Name', 'Total Attendance']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}'. Found columns: {df.columns.tolist()}")
    return df

# =======================================
# MARK HOLIDAYS OR WEEKENDS AS "Off"
# =======================================
def mark_off_if_needed(excel_file_path):
    today_date = datetime.now().strftime('%d-%m-%Y')
    weekday = datetime.now().weekday()  # Monday=0, Sunday=6
    
    if weekday >= 5 or today_date in HOLIDAYS:
        df = pd.read_excel(excel_file_path)
        df = clean_columns(df)

        status_col = f'{today_date} Status'
        time_col = f'{today_date} Time'

        if status_col not in df.columns:
            df[status_col] = 'Off'
            df[time_col] = ''

            df.to_excel(excel_file_path, index=False)
            print(f"🛑 Marked 'Off' in {excel_file_path} for {today_date}.")
            return True
        else:
            print(f"ℹ️ Already marked 'Off' in {excel_file_path} for {today_date}.")
            return True
    return False

# =======================================
# MARK ALL ABSENTEES BEFORE ATTENDANCE
# =======================================
def mark_absentees_before_attendance(excel_file_path):
    today_date = datetime.now().strftime('%d-%m-%Y')
    df = pd.read_excel(excel_file_path)
    df = clean_columns(df)

    status_col = f'{today_date} Status'
    time_col = f'{today_date} Time'

    if status_col not in df.columns:
        df[status_col] = ''
        df[time_col] = ''

    updated = False

    for index in df.index:
        if df.at[index, status_col] == '':
            df.at[index, status_col] = 'A'
            df.at[index, time_col] = '00:00:00'
            updated = True
            print(f"❌ Marked Absent in {excel_file_path}: ID {df.at[index, 'Student ID']}")

    if updated:
        df.to_excel(excel_file_path, index=False)
        print(f"✅ Absentees marked in {excel_file_path}.")
    else:
        print(f"ℹ️ No absentees in {excel_file_path}.")

# ===========================================
# MARK A STUDENT PRESENT IN EXCEL ATTENDANCE
# ===========================================
def get_excel_path(student_id):
    prefix = str(student_id)[:2]
    if prefix == '41':
        return 'C:/Face-Detection-System/stud_data/STEP 1.xlsx'
    elif prefix == '42':
        return 'C:/Face-Detection-System/stud_data/STEP 2.xlsx'
    elif prefix == '43':
        return 'C:/Face-Detection-System/stud_data/STEP 3.xlsx'
    else:
        raise ValueError(f"Invalid student ID: {student_id}")

def MarkAttendance(student_id):
    current_time = datetime.now()
    current_time_str = current_time.strftime("%I:%M:%S %p")
    current_time_time = current_time.time()

    # Check if current time is within attendance window
    if not (ATTENDANCE_START_TIME <= current_time_time <= ATTENDANCE_END_TIME):
        print(f"⚠️ Attendance time closed for ID {student_id} at {current_time_str}")
        return

    today_date = datetime.now().strftime('%d-%m-%Y')
    excel_path = get_excel_path(student_id)
    df = pd.read_excel(excel_path)
    df = clean_columns(df)

    status_col = f'{today_date} Status'
    time_col = f'{today_date} Time'

    if status_col not in df.columns:
        df[status_col] = ''
        df[time_col] = ''

    row_index = df.index[df['Student ID'] == int(student_id)].tolist()

    if not row_index:
        print(f"⚠️ Student ID {student_id} not found in {excel_path}.")
        return

    idx = row_index[0]

    if df.at[idx, status_col] != 'P':
        df.at[idx, status_col] = 'P'
        df.at[idx, time_col] = current_time_str  # Store in 12-hour format

        total_attendance = df.at[idx, 'Total Attendance']
        df.at[idx, 'Total Attendance'] = int(total_attendance) + 1 if str(total_attendance).isdigit() else 1

        df.to_excel(excel_path, index=False)
        print(f"✅ Marked Present in {excel_path}: ID {student_id} | Time: {current_time_str}")
    else:
        print(f"ℹ️ Already Present in {excel_path}: ID {student_id}")

def process_detection(student_id):
    global last_seen
    current_time = time.time()

    if student_id in last_seen:
        elapsed_time = current_time - last_seen[student_id]
        if elapsed_time < COOLDOWN_SECONDS:
            return
    
    last_seen[student_id] = current_time
    MarkAttendance(student_id)

# =====================================
# LOAD STUDENT NAME FROM EXCEL
# =====================================
def LoadStudentName(student_id):
    excel_path = get_excel_path(student_id)
    df = pd.read_excel(excel_path)
    df = clean_columns(df)

    student_row = df[df['Student ID'] == int(student_id)]
    if not student_row.empty:
        total_attendance = student_row.iloc[0]['Total Attendance']
        return student_row.iloc[0]['Name'], total_attendance
    return f'Student {student_id}', 0

# =====================================
# MAIN SYSTEM EXECUTION
# =====================================
def main():
    global current_mode, mode_timeout
    current_mode = 0
    mode_timeout = 0
    detected_students = set()
    cooldown_tracker = {}
    student_data = {}
    
    # Initialize camera
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not access the camera.")
        return

    cap.set(3, FRAME_WIDTH)
    cap.set(4, FRAME_HEIGHT)
    time.sleep(2)

    # File operations
    step_files = [
        'C:/Face-Detection-System/stud_data/STEP 1.xlsx',
        'C:/Face-Detection-System/stud_data/STEP 2.xlsx',
        'C:/Face-Detection-System/stud_data/STEP 3.xlsx'
    ]

    # Holiday check
    is_holiday = False
    for file_path in step_files:
        if mark_off_if_needed(file_path):
            is_holiday = True

    if is_holiday:
        print("🚫 System exiting due to holiday/weekend.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # Mark absentees
    for file_path in step_files:
        mark_absentees_before_attendance(file_path)

    print("📸 Starting Face Detection... Press 'q' to exit.\n")
    mode_start_time = None

    while True:
        try:
            success, img = cap.read()
            if not success:
                print("⚠️ Frame not captured, skipping...")
                time.sleep(0.1)
                continue

            # Face Processing Pipeline
            stud_img_size = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            stud_img_size = cv2.cvtColor(stud_img_size, cv2.COLOR_BGR2RGB)

            face_current_frame = face_recognition.face_locations(stud_img_size)
            encode_current_frame = face_recognition.face_encodings(stud_img_size, face_current_frame)

            # Update UI background
            frame_bg = imagebackground.copy()
            frame_bg[162:162+480, 55:55+640] = img
            current_mode = max(0, min(current_mode, 3))
            
            # MODE 1 DISPLAY SECTION
            if current_mode == 1:
                # Apply Mode 1 background
                frame_bg[44:44+633, 808:808+414] = imgModeList[1]
                
                if student_id in student_data:
                    # Student image
                    if student_data[student_id]['image'] is not None:
                        frame_bg[175:175+216, 909:909+216] = student_data[student_id]['image']

                    # Attendance counter
                    cv2.putText(frame_bg, 
                                str(student_data[student_id]['attendance']), 
                                (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX, 
                                1, 
                                (255, 255, 255), 
                                1)

                    # Student name
                    (w, h), _ = cv2.getTextSize(student_data[student_id]['name'], 
                                              cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    center_offset = (414 - w) // 2
                    cv2.putText(frame_bg, 
                                student_data[student_id]['name'], 
                                (808 + center_offset, 445), 
                                cv2.FONT_HERSHEY_COMPLEX, 
                                1, 
                                (50, 50, 50), 
                                1)

                    # Student ID
                    cv2.putText(frame_bg, 
                                f"ID: {student_id}", 
                                (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 
                                0.5, 
                                (255, 255, 255), 
                                1)
            else:
                frame_bg[44:44+633, 808:808+414] = imgModeList[current_mode]

            # Handle mode timeout
            if current_mode == 1:
                elapsed_time = time.time() - mode_start_time
                if elapsed_time >= MODE_DISPLAY_DURATION:
                    current_mode = 0

            current_time = datetime.now()
            face_detected = False

            # Process each face
            for encode_face, face_location in zip(encode_current_frame, face_current_frame):
                if not founded_encodings:
                    print("⚠️ No known encodings available.")
                    continue

                matches = face_recognition.compare_faces(founded_encodings, encode_face, FACE_MATCH_THRESHOLD)
                face_distances = face_recognition.face_distance(founded_encodings, encode_face)
                
                if len(face_distances) == 0:
                    continue

                match_index = np.argmin(face_distances)

                if matches[match_index]:
                    face_detected = True
                    student_id = stud_ID[match_index]
                    
                    # Draw rectangle and text
                    top, right, bottom, left = [v * 4 for v in face_location]
                    bg_left = 55 + left
                    bg_top = 162 + top
                    bg_right = 55 + right
                    bg_bottom = 162 + bottom
                    
                    cv2.rectangle(frame_bg, (bg_left, bg_top), (bg_right, bg_bottom), (0, 255, 0), 2)
                    cv2.putText(frame_bg, f"ID: {student_id}", (bg_left, bg_top - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Cooldown check
                    last_seen = cooldown_tracker.get(student_id)
                    if last_seen and (datetime.now() - last_seen).total_seconds() < COOLDOWN_MINUTES * 60:
                        print(f"⏳ Already marked: {student_id}")
                        current_mode = 3
                        mode_timeout = datetime.now().timestamp() + MODE_DISPLAY_DURATION
                        continue

                    # Update records
                    if student_id not in detected_students:
                        process_detection(student_id)
                        detected_students.add(student_id)

                    cooldown_tracker[student_id] = current_time
                    current_mode = 1
                    mode_start_time = time.time()

                    # Load student data
                    if student_id not in student_data:
                        name, attendance = LoadStudentName(student_id)
                        student_data[student_id] = {
                            'name': name,
                            'attendance': attendance,
                            'image': cv2.imread(f'C:/Face-Detection-System/resized_images/{student_id}.jpg')
                        }

            # Update detection mode
            if not face_detected and current_mode not in [2, 3]:
                current_mode = 0
            elif face_detected and current_mode == 0:
                current_mode = 1
                mode_timeout = datetime.now().timestamp() + MODE_DISPLAY_DURATION

            # Show frame
            cv2.imshow("Face-Attendance", frame_bg)

            # Exit handling
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                print("🛑 User requested exit. Exiting...")
                break
            
            if cv2.getWindowProperty("Face-Attendance", cv2.WND_PROP_VISIBLE) < 1:
                print("🛑 Window closed by user. Exiting...")
                break

        except Exception as e:
            print(f"❌ Critical error: {str(e)}")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()