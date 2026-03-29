from pipelines.inference_image import image_model
import cv2
import os
import numpy as np
from PIL import Image
class video_model():
    def __init__(self):
        self.model=image_model()
    def predict(self,video):
        self.lis=[]
        self.video_path=video
        cap=cv2.VideoCapture(self.video_path)
        file_name=os.path.splitext(os.path.basename(self.video_path))[0]
        max_frame=30
        frame_interval=3
        frame_count=0
        while True:
            success,frame=cap.read()
            if not success:
                break
            if frame_count % frame_interval==0 and frame_count<=max_frame:
                img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                self.lis.append(self.model.predict(img))
            frame_count+=1
        cap.release()    
        real_count=0
        fake_count=0
        for i in self.lis:
            if i=="Real":
                real_count+=1
            elif i=='Fake':
                fake_count+=1
        response= "Real" if real_count>fake_count else "Fake"
        os.rename(self.video_path,f'output/video/{file_name}_{response}.mp4')
        return response
        


            






