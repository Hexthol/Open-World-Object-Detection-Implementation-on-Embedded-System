import cv2
import argparse
from YOLOv8_ONNX import YOLOv8, YOLOv8Seg, YOLOv8Pose
from gtts import gTTS
from playsound import playsound
import os
import time
from threading import Thread


# yolo = YOLOv8Seg("models/yolov8n-seg.onnx", conf_thres=0.3, iou_thres=0.5)
# yolo = YOLOv8Pose("models/yolov8n-pose.onnx", conf_thres=0.3, iou_thres=0.5)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8Live")
    parser.add_argument(
        "--webcam-resolution",
        default=[960,960],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def text_to_speech(text, mp3_file_path='output.mp3'):
    # Create a gTTS object
    tts = gTTS(text=text, lang='en')

    # Save the converted audio to an MP3 file
    tts.save(mp3_file_path)

def play_mp3(mp3_file_path):
    playsound(mp3_file_path)

def thread_func(class_ids, class_names):

    text = "You still need "
    temp = True
    for key in class_names:
        if key != 11 and key not in class_ids:
            temp = False
            text = text + class_names[key] + ", "
    text = text + ". "

    if temp:
        text = "You got every required item. "

    if 11 in class_ids:
        text = text + "Distractors also detected. Please remove it."

    text_to_speech(text, 'output.mp3')
    play_mp3('output.mp3')

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    yolo = YOLOv8("best_new_960.onnx", conf_thres=0.3, iou_thres=0.5)

    cap = cv2.VideoCapture(0)
    mp3_file_path = 'output.mp3'
    class_names = {0: 'measuring spoons', 1: 'small spoon', 2: 'bowl', 3: 'glass measuring cup', 4: 'timer', 5: 'salt', 6: 'hot pad', 7: 'one over two measuring cup', 8: 'pan', 9: 'oatmeal', 10: 'big spoon', 11: 'distractors'}

    frame_rate = 1
    prev = 0
    cue_interval = 30
    prev1 = 0

    while True:
        time_elapsed = time.time() - prev
        ret, frame = cap.read()
        if time_elapsed > 1./frame_rate:
            prev = time.time()
            yolo(frame)
            combined_img = yolo.draw_results(frame)
            cv2.imshow("Detected Objects", combined_img)
            print(yolo.class_ids)
            if time.time() - prev1 > cue_interval:
                prev1 = time.time()
                t = Thread(target = thread_func, args = (yolo.class_ids, class_names, ))
                print("One audio coming")
                t.start()

            
        k = cv2.waitKey(2)
        if k != -1:
            break
    
    os.remove(mp3_file_path)
        

if __name__ == "__main__":
    main()

