import cv2
import os
import pickle
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ================================
# CONFIGURATION & GLOBAL SETTINGS
# ================================
ENCODING_FILE = 'Encoding_File.p'

HOLIDAYS = ['23-03-2025', '01-05-2025', '14-08-2025']  # Add more dates as needed
COOLDOWN_MINUTES = 5
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

# ========================
# INITIALIZE VIDEO STREAM
# ========================
cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

# ========================
# LOAD UI RESOURCES
# ========================
imagebackground = cv2.imread('Resources/background.png')
modeFolderPath = 'Resources/Modes'
modePath = os.listdir(modeFolderPath)
imgModeList = [cv2.imread(os.path.join(modeFolderPath, path)) for path in modePath]

# ========================
# LOAD FACE ENCODINGS
# ========================
with open(ENCODING_FILE, 'rb') as encode_file:
    founded_encodings, stud_ID = pickle.load(encode_file)

# ========================
# TRACKERS & CACHES
# ========================
detected_students = set()
cooldown_tracker = {}
student_data = {}

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
    today_date = datetime.now().strftime('%d-%m-%Y')
    current_time = datetime.now().strftime('%H:%M:%S')

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
        df.at[idx, time_col] = current_time

        total_attendance = df.at[idx, 'Total Attendance']
        df.at[idx, 'Total Attendance'] = int(total_attendance) + 1 if str(total_attendance).isdigit() else 1

        df.to_excel(excel_path, index=False)
        print(f"✅ Marked Present in {excel_path}: ID {student_id} | Time: {current_time}")
    else:
        print(f"ℹ️ Already Present in {excel_path}: ID {student_id}")

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
    modeType = 0
    counter = 0

    step_files = [
        'C:/Face-Detection-System/stud_data/STEP 1.xlsx',
        'C:/Face-Detection-System/stud_data/STEP 2.xlsx',
        'C:/Face-Detection-System/stud_data/STEP 3.xlsx'
    ]

    # Check holidays for all files
    is_holiday = False
    for file_path in step_files:
        if mark_off_if_needed(file_path):
            is_holiday = True

    if is_holiday:
        print("🚫 System exiting due to holiday/weekend.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # Mark absentees in all files
    for file_path in step_files:
        mark_absentees_before_attendance(file_path)

    print("📸 Starting Face Detection... Press 'q' to exit.\n")

    while True:
        success, img = cap.read()
        stud_img_size = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        stud_img_size = cv2.cvtColor(stud_img_size, cv2.COLOR_BGR2RGB)

        face_current_frame = face_recognition.face_locations(stud_img_size)
        encode_current_frame = face_recognition.face_encodings(stud_img_size, face_current_frame)

        imagebackground[162:162+480, 55:55+640] = img
        imagebackground[44:44+633, 808:808+414] = imgModeList[modeType]

        current_time = datetime.now()

        for encode_face, face_location in zip(encode_current_frame, face_current_frame):
            matches_faces = face_recognition.compare_faces(founded_encodings, encode_face)
            faces_distance = face_recognition.face_distance(founded_encodings, encode_face)

            match_index = np.argmin(faces_distance)

            if matches_faces[match_index]:
                student_id = stud_ID[match_index]
                print(f"👀 Detected ID: {student_id}")

                last_detected_time = cooldown_tracker.get(student_id)
                if last_detected_time and (current_time - last_detected_time < timedelta(minutes=COOLDOWN_MINUTES)):
                    print(f"⏳ Cooldown for {student_id}.")
                    continue

                cooldown_tracker[student_id] = current_time

                if student_id not in detected_students:
                    detected_students.add(student_id)
                    MarkAttendance(student_id)

                if student_id not in student_data:
                    student_name, total_attendance = LoadStudentName(student_id)
                    student_data[student_id] = {'name': student_name}

                modeType = 1
                counter = 1
                student_image_path = os.path.join('C:/Face-Detection-System/resized_images/', f'{student_id}.jpg')
                # loading image to display on  Screen
                if os.path.exists(student_image_path):
                    student_img = cv2.imread(student_image_path)
                    imagebackground[175:175+216, 909:909+216] = student_img

        if counter != 0:
            if counter == 1:
                print(f"🎉 Displaying {student_data[student_id]['name']}.")
                # top right corner
            cv2.putText(imagebackground, f'{total_attendance}', (861, 125), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            # displaying Student name in Excel to Screen Center
            (w, h), _ = cv2.getTextSize(student_data[student_id]['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            center = (414 - w) // 2
            cv2.putText(imagebackground, student_data[student_id]['name'], (808 + center, 445), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)
            #displaying Student-ID at ID 
            cv2.putText(imagebackground, f"ID: {student_id}", (1006, 550), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            #displaying Student-ID at Major
            cv2.putText(imagebackground, f"ID: {student_id}", (1006, 493), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            # cv2.putText(imagebackground,
            #             student_data[student_id]['standing'],
            #             (910, 625),
            #             cv2.FONT_HERSHEY_COMPLEX,
            #             1,
            #             (255, 255, 255),
            #             1)
            
            # cv2.putText(imagebackground,
            #             student_data[student_id]['yaer'],
            #             (1025,625 ),
            #             cv2.FONT_HERSHEY_COMPLEX,
            #             1,
            #             (255, 255, 255),
            #             1)
            
            # cv2.putText(imagebackground,
            #             student_data[student_id]['Starting Year'],
            #             (808, 445),
            #             cv2.FONT_HERSHEY_COMPLEX,
            #             1,
            #             (255, 255, 255),
            #             1)
            # Displaying Student Image who is detected
            imagebackground[175:175+216,909:909+216] = student_img
            counter += 1
            if counter > 10:
                modeType = 0
                counter = 0

        cv2.imshow("Face-Attendance", imagebackground)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            print("🛑 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

# ===================
# EXECUTE MAIN
# ===================
if __name__ == "__main__":
    main()