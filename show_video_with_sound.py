import numpy as np
import cv2
from pydub import AudioSegment
from pydub.playback import play
from threading import Thread
from multiprocessing import Process, Value





def play_video():
    cap = cv2.VideoCapture("vid_example.mp4")
    while cap.isOpened():
        ret, fram = cap.read()

        if ret:
            gray = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
            cv2.imshow('video', gray)
            cv2.waitKey(1)

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def play_sound(on=True):
    if on:
        song = AudioSegment.from_mp3("Thunder.mp3")
        play(song)







class play_sound_with_cond():

    def __init__(self):
        self.sound = False
        self.cap = cv2.VideoCapture("vid_example.mp4")




    def show_frame(self):
        ret, fram = self.cap.read()
        if ret:
            gray = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
            cv2.imshow('video', gray)

            if cv2.waitKey(100) & 0xFF == ord('e'):
                self.sound = True



    def play_sound(self):
        if self.sound:
            song = AudioSegment.from_mp3("Thunder.mp3")
            play(song)
        else:
            print("no inputs")






if __name__ == "__main__":

    prog = play_sound_with_cond()
    th1 = Thread(target=play_video)
    th2 = Thread(target=play_sound)
    #
    # # th1 = Thread(target=play_video)
    # # th2 = Thread(target=play_sound)
    #
    th1.start()
    th2.start()
    th1.join()
    th2.join()

