import streamlit as st
import torch
import cv2
import tempfile
import os
from PIL import Image
import numpy as np
from ultralytics import YOLO
import time

from RT.models.utils import blob
from RT.models.torch_utils import det_postprocess
from RT.models.cudart_api import TRTEngine
from RT.models import EngineBuilder

class Deployment:
    def __init__(self):
        self.models = ["best.pt","best.onnx","best.engine"]
        self.colors = [(255,0,0),(0,0,255),(0,0,0)]
        

    def ImageBox(self, image):
        
        new_shape=(640, 640)
        width, height, channel = image.shape
        
        ratio = min(new_shape[0] / width, new_shape[1] / height)
        new_unpad = int(round(height * ratio)), int(round(width * ratio))
        dw, dh = (new_shape[0] - new_unpad[0])/2, (new_shape[1] - new_unpad[1])/2

        if (height, width) != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = self.colors[-1])
        
        return image, ratio, (dw, dh)


    def face_detection_for_image_with_pt(self, file):
        start = time.time()
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes,1)
        img = np.array(img)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        model = YOLO(self.models[0])
        results = model.predict(image, conf=0.7, stream = True,device = "cuda:0")

        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            for box in boxes:
                points = box.xyxy[0].astype(int)

                cv2.rectangle(image, points[:2], points[2:], self.colors[0], 2)
        end = time.time()

        return image, 1.0/(end-start)



    def face_detection_for_image_with_engine(self, file):
        start = time.time()
        image = None
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(file.read())

            image = cv2.imread(temp_filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        model = TRTEngine(self.models[-1])
        
        img, ratio, dwdh = self.ImageBox(image)
        tensor = blob(img, return_seg=False)
        tensor = torch.asarray(tensor)

        dwdh = np.array(dwdh * 2, dtype=np.float32)

        results = model(tensor)

        bboxes, scores, labels = det_postprocess(results)
        bboxes = (bboxes-dwdh)/ratio
        
        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().astype(np.int32).tolist()
            cv2.rectangle(image, (bbox[0],bbox[1]) , (bbox[2],bbox[3]) , self.colors[1], 2)
        end = time.time()
        os.remove(temp_filename)
        
        return image, 1.0/(end-start)


    
    def face_detection_for_video_with_pt(self, video_path):

        video_path = None
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            video_path = temp_file.name
            temp_file.write(file.read())

        model = YOLO(self.models[0])
        
        FRAME_WINDOW = st.image([])
        
        video = cv2.VideoCapture(video_path)
        st.write("video started")
        while video.isOpened():
            try:
                
                ret, frame = video.read()
                start = time.time()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model.predict(frame, conf=0.7, stream = True,device = "cuda:0")
                
                for result in results:
                    boxes = result.boxes.cpu().numpy()  
                    for box in boxes:
                        points = box.xyxy[0].astype(int)
                        cv2.rectangle(frame, points[:2], points[2:], self.colors[0], 2)
                end = time.time()
                cv2.putText(frame,str(end-start)+" sec",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors[-1], 2, cv2.LINE_AA)
                
                if not ret or video.isOpened()==False:
                    break

                FRAME_WINDOW.image(frame)
                
            except:
                break
        
        os.remove(video_path)
        video.release()
        st.write("video ended")


        
    def face_detection_for_video_with_engine(self, file):
        
        video_path = None
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            video_path = temp_file.name
            temp_file.write(file.read())

        model = TRTEngine(self.models[-1])
        FRAME_WINDOW = st.image([])
        
        video = cv2.VideoCapture(video_path)
        st.write("video started")
        while video.isOpened():
            try:
                ret, frame = video.read()
                start = time.time()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                start = time.time()
                image, ratio, dwdh = self.ImageBox(frame)

                tensor = blob(image, return_seg=False)
                tensor = torch.asarray(tensor)

                results = model(tensor)
                dwdh = np.array(dwdh * 2, dtype=np.float32)

                bboxes, scores, labels = det_postprocess(results)
                bboxes = (bboxes-dwdh)/ratio
                end = time.time()
                
                for (bbox, score, label) in zip(bboxes, scores, labels):
                    bbox = bbox.round().astype(np.int32).tolist()
                    cv2.rectangle(frame, (bbox[0],bbox[1]) , (bbox[2],bbox[3]) , self.colors[1], 2)
                    
                end = time.time()
                cv2.putText(frame,str(end-start)+" sec",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors[-1], 2, cv2.LINE_AA)
                if not ret or video.isOpened()==False:
                    break

                FRAME_WINDOW.image(frame)

            except:
                break
            
        os.remove(video_path)
        video.release()
        st.write("video ended")


if  __name__ == '__main__':
    st.title("Deployment Face Detection model with PyTorch and TensorRT")
    
    model = st.selectbox('Select model type:', ('PyTorch (*.pt)', 'TensorRT (*.engine)'))
    source = st.selectbox('Select source type:', ('video', 'image'))

    file = st.file_uploader("Upload your file here...", type=["mp4", "avi", "mov", "jpg", "png", "jpeg"])

    deployment = Deployment()
    
    if file is not None:
        col1, col2 = st.columns(2)
        
        if source == "image":
            if model=="PyTorch (*.pt)":
                st.write("Just a minute image predicted...")
                predicted_image, fps = deployment.face_detection_for_image_with_pt(file)
                info = {"Keys":["model name","format","size","type","fps"],"Values":["best.pt","*.pt","50 763 KB","fp16",fps]}
                col1.image(predicted_image)
                col2.dataframe(info,use_container_width=True)

            elif model == "TensorRT (*.engine)":
                st.write("Just a minute image predicted...")
                predicted_image, fps = deployment.face_detection_for_image_with_engine(file)
                info = {"Keys":["model name","format","size","type","fps"],"Values":["best.engine","*.engine","53 763 KB","fp16",fps]}
                col1.image(predicted_image)
                col2.dataframe(info,use_container_width=True)


        else:
            if model=="PyTorch (*.pt)":
                deployment.face_detection_for_video_with_pt(file)
                
            elif model == "TensorRT (*.engine)":
                deployment.face_detection_for_video_with_engine(file)
