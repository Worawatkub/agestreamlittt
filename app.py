import cv2 #เรียกใช้ open cv
import time
import streamlit as st #เรียกใช้ streamlit
import pandas as pd #เรียกใช้ pandas
import numpy as np #เรียกใช้ numpy
from PIL import Image
from datetime import datetime

class Detectface(): #เขียนอยู่ในรูปแบบ oop โดยการเรียกใช้ฟังก์ชั่น
    def get_face_box(net, frame, conf_threshold=0.7): #โค๊ด Python สำหรับการตรวจจับใบหน้า ตั้งแต่ 9-29
        opencv_dnn_frame = frame.copy()
        frame_height = opencv_dnn_frame.shape[0]
        frame_width = opencv_dnn_frame.shape[1]
        blob_img = cv2.dnn.blobFromImage(opencv_dnn_frame, 1.0, (300, 300), [104, 117, 123], True, False) #เรียกใช้อัลกอริทึม blob = cv2.dnn.blobFromImage และกำหนด scale factor ตามที่ต้องการ

        net.setInput(blob_img)
        detections = net.forward()
        b_boxes_detect = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                b_boxes_detect.append([x1, y1, x2, y2])
                cv2.rectangle(opencv_dnn_frame, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frame_height / 150)), 8)
        return opencv_dnn_frame, b_boxes_detect

st.write("# Age and Gender prediction"+":star2:")
st.write("## Upload a picture that contains a face"+":camera:")

bytes_data = None
t=time.time()
input_mode = st.radio("# Input mode", ["Camera", "File upload"]) #กำหนกให้สามารถเลือกได้ว่าจะใช้ กล้อง หรือ อัพโหลดรูปภาพ

if input_mode == "Camera": #เรียกใช้กล้อง
    img_file_buffer = st.camera_input("Take a picture")
    photo = img_file_buffer # กำหนดตัวแปลให้ photo เป็น img_file_buffer
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
else:
    uploaded_file = st.file_uploader("โปรดอัพรูปของคุณ", type=['png', 'jpg'])
    photo = uploaded_file # กำหนดตัวแปลให้ photo เป็น uploaded_file
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
if bytes_data is None:
    st.stop()
    
if photo:
    image = Image.open(photo) #เรียกใช้ photo
    cap = np.array(image)
    cv2.imwrite('temp.jpg', cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY))
    cap=cv2.imread('temp.jpg')
    #69-76 เรียกใช้ข้อมูลมาเก็บไว้ในตัวแแปล เพื่อให้ใช้งานได้สะดวกขึ้น
    face_txt_path="opencv_face_detector.pbtxt"
    face_model_path="opencv_face_detector_uint8.pb"

    age_txt_path="age_deploy.prototxt" #ไฟล์ที่ลงท้ายด้วย .prototxt ไฟล์นี้กำหนดเลเยอร์ในโครงข่ายประสาทเทียม อินพุต เอาต์พุต และฟังก์ชันของแต่ละเลเยอร์ 72 และ 75 #ไฟล์ CNN ที่ช่วย แยกอายุ
    age_model_path="age_net.caffemodel" #ไฟล์ที่ลงท้ายด้วย .caffemodel 73 และ 76 คือ This contains the information of the trained neural network (trained model)

    gender_txt_path="gender_deploy.prototxt" #ไฟล์ CNN ที่ช่วย แยกเพศ
    gender_model_path="gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    age_classes=['Age: ~1-2', 'Age: ~3-5', 'Age: ~6-14', 'Age: ~16-22',
                'Age: ~25-30', 'Age: ~32-40', 'Age: ~45-50', 'Age: age is greater than 60'] #แบ่งช่วงอายุทั้งหมด 8 คลาสตามความน่าจะเป็น
    gender_classes = ['Male', 'Female']

    age_net = cv2.dnn.readNet(age_model_path, age_txt_path) #cv2.dnn.readNet ทำการโหลดอัลกอริทึม จากไฟล์ age_net.caffemodel และ age_deploy.prototxt มาเก็บใน ageNet
    gender_net = cv2.dnn.readNet(gender_model_path, gender_txt_path) #cv2.dnn.readNet ทำการโหลดอัลกอริทึม จากไฟล์ gender_net.caffemodel และ gender_deploy.prototxt มาเก็บใน genderNet
    face_net = cv2.dnn.readNet(face_model_path, face_txt_path) #cv2.dnn.readNet ทำการโหลดอัลกอริทึม จากไฟล์ opencv_face_detector_uint8.pb และ opencv_face_detector.pbtxt มาเก็บใน faceNet

    padding = 20 #Padding กำหนดขอบในที่นี้กำหนดให้เท่ากับ 20 Pixels
    t = time.time()
    frameFace, b_boxes = Detectface.get_face_box(face_net, cap) # Detectface เรียกใช้ฟังก์ชั่น highlightFace นอกคลาส
    
    if not b_boxes:
        st.write("ไม่พบใบหน้าที่กำลังตรวจสอบ")

    for bbox in b_boxes: #กรอบใบหน้า
        face = cap[max(0, bbox[1] -
                    padding): min(bbox[3] +
                                    padding, cap.shape[0] -
                                    1), max(0, bbox[0] -
                                            padding): min(bbox[2] +
                                                        padding, cap.shape[1] -
                                                        1)]

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob) #ทำนายเพศที่คาดการ ตั้งแต่บรรทัดที่ 93-97
        gender_pred_list = gender_net.forward()
        gender = gender_classes[gender_pred_list[0].argmax()]
        st.write(
            f"Gender : {gender}, confidence = {gender_pred_list[0].max() * 100}%")

        age_net.setInput(blob) #ทำนายอายุที่คาดการ ตั้งแต่บรรทัดที่ 101-104
        age_pred_list = age_net.forward()
        age = age_classes[age_pred_list[0].argmax()]
        st.write(f"Age : {age}, confidence = {age_pred_list[0].max() * 100}%")

        label = "{},{}".format(gender, age)
        cv2.putText(
            frameFace,
            label,
            (bbox[0],
            bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,
            255,
            255),
            2,
            cv2.LINE_AA) #cv2.putTextระบุตำแน่งอักษรหรือข้อความที่จะเขียน

        st.image(frameFace) #้โชว์รูปผ่านเว็ป

        st.sidebar.header("บันทึกข้อมูลลงในฐานข้อมูล")
        df = pd.read_csv("data/test.csv") #กำหนดเส้นทางให้ df
        st.sidebar.table(df) #โชว์ตารางออกมา
        options_form = st.sidebar.form("options_form") #กำหนดตัวเลือก
        add_name = options_form.text_input("โปรดใส่ชื่อของคุณ") #กำหนดกล่องข้อความให้รับค่า ชื่อ
        now = datetime.now() #เซ็ตค่าให้เป็นเวลาปัจจุบัน
        user_time = now.strftime("%H:%M:%S") #กำหนด ชั่วโมง นาที วินาที
        add_data = options_form.form_submit_button("บันทึกจัดเก็บข้อมูล") #กำหนดปุ่มบันทึก
        
        if add_data is not None: #กำหนดให้ทำการบันทึกข้อมูลไปยัง CSV
            new_data = {"name": add_name , "gender": gender ,"age": age ,"time": user_time} #กำหนดให้บันทึกคอลัมน์
            df = df.append(new_data, ignore_index = True)
            df.to_csv("data/test.csv" , index = False)

        if add_data: #กำหนดให้แสดงข้อความเมื่อกดปุ่ม
            st.sidebar.header("บันทึกข้อมูลสำเร็จ") #แสดงข้อความทางแถบซ้าย

            im_pil = Image.fromarray(frameFace)
            im_pil.save('Result.jpeg') #บันทึกรูปผลลัพธ์ลงได้ฐานข้อมูล
            with open("Result.jpeg", "rb") as file:  # เปิดรูป Result.jpeg เพื่อใช้ในการกำหนดปุ่มโหลดภาพ
                btn = st.download_button( #กำหนดปุ่มโหลดภาพ
                        label="ดาวน์โหลดรูปภาพ", #กำหนดตัวอักษรในปุ่ม
                        data=file, #กำหนดให้ data มีค่าเท่ากับ file
                        file_name="image_test.jpeg", #กำหนดชื่อภาพ
                    )
            if btn :
                st.write('ดาวน์โหลดสำเร็จ')
            else :
                st.write("คุณยังไม่ได้ทำการดาวน์โหลดรูปภาพ")
        if add_data :
            st.write('สรุปกราฟ เพศและอายุ ของผู้ใช้งานทั้งหมด')
            df2 = df.filter(regex= '(gender|age)', axis=1) #กำหนดให้แสดงแค่ เพศและอายุ
            st.line_chart(df2) #กำหนดให้แสดงกราฟ