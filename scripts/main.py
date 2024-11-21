import mediapipe as mp
import numpy as np
import pandas as pd
from threading import Thread
from datetime import datetime
import cv2
import time
import pygame
import glob
import os
import random

# Initialize width and height camera
cameraWidth = 1280
cameraHeight = 720

# Define index of tips position
tipsId = [4, 8, 12, 16, 20]

# Initialize note names
whiteNotes = ["4-c", "4-d", "4-e", "4-f", "4-g", "4-a", "4-b",
              "5-c", "5-d", "5-e", "5-f", "5-g", "5-a", "5-b"]

blackNotes = ["4-cs", "4-ds", "4-fs", "4-gs", "4-as",
              "5-cs", "5-ds", "5-fs", "5-gs", "5-as"]

# Initialize settings for white key
widthWhiteNoteKey = 65
heightWhiteNoteKey = 330
shiftWhiteNote = 150
whiteColor = [255, 255, 255]

# Initialize settings for black key
widthBlackNoteKey = int(0.7 * widthWhiteNoteKey)
heightBlackNoteKey = int(2 / 3 * heightWhiteNoteKey)
shiftBlackNote = 150 + int(0.7 * widthWhiteNoteKey)
blackColor = [0, 0, 0]

buttonList = []
processDictionary = {}

class HandDetector:

    def __init__(self, mode=False, maxNumberHands=2, complexity=0, detectionConfidence=0.6, trackConfidence=0.5):
        self.mode = mode
        self.maxNumberHands = maxNumberHands
        self.complexity = complexity
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        self.results = None

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxNumberHands, self.complexity, self.detectionConfidence,
                                        self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def detectHands(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Mediapipe model works only with RGB mode
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        self.results = self.hands.process(image)  # Hand landmark detection process
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.drawHandsConnections(image)

        return image

    def drawHandsConnections(self, image):
        if self.results.multi_hand_landmarks:
            for handIndex, coordinatesLandmark in enumerate(self.results.multi_hand_landmarks):
                self.mpDraw.draw_landmarks(image, coordinatesLandmark, self.mpHands.HAND_CONNECTIONS,
                                           self.mpDraw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=4),
                                           self.mpDraw.DrawingSpec(color=(157, 168, 58), thickness=3, circle_radius=4),
                                           )
                if self.showHandLabel(handIndex, coordinatesLandmark, self.results):
                    label, coordinates = self.showHandLabel(handIndex, coordinatesLandmark, self.results)
                    cv2.putText(image, label, coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    def showHandLabel(self, index, coordinates, result):
        label = result.multi_handedness[index].classification[0].label
        coordinates = tuple(np.multiply(
            np.array(
                (coordinates.landmark[self.mpHands.HandLandmark.WRIST].x,
                 coordinates.landmark[self.mpHands.HandLandmark.WRIST].y)),
            [cameraWidth, cameraHeight]).astype(int))

        return label, coordinates

    def findLandmarkList(self, img):
        landmarkList = []

        if self.results.multi_hand_landmarks:
            for handIndex, coordinates in enumerate(self.results.multi_hand_landmarks):
                label = self.results.multi_handedness[handIndex].classification[0].label
                myHand = self.results.multi_hand_landmarks[handIndex]
                handLandmark = []
                for index, lm in enumerate(myHand.landmark):
                    height, width, channels = img.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    handLandmark.append([index, cx, cy, label])
                landmarkList.append(handLandmark)

        return landmarkList


class Button:

    def __init__(self, name, position, color, sound, size):
        self.name = name
        self.position = position
        self.color = color
        self.sound = sound
        self.size = size


class SoundPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.sounds = {}
        self.load_sounds()

    def load_sounds(self):
        soundsList = defineSoundTrackList()
        music_dir = os.path.join(os.getcwd(), "music")
        for sound in soundsList:
            path = os.path.join(music_dir, sound)
            if os.path.exists(path):
                try:
                    self.sounds[sound] = pygame.mixer.Sound(path)
                except pygame.error as e:
                    print(f"Cannot load sound: {path} - {e}")
            else:
                print(f"Sound file does not exist: {path}")

    def play_sound(self, sound):
        if sound in self.sounds:
            self.sounds[sound].play()
        else:
            print(f"Sound not loaded: {sound}")

    def stop_sound(self, sound):
        if sound in self.sounds:
            self.sounds[sound].stop()
        else:
            print(f"Sound not loaded: {sound}")


def setCaptureDeviceSetting(cameraID=0):
    camera = cv2.VideoCapture(cameraID, cv2.CAP_DSHOW)
    camera.set(3, cameraWidth)
    camera.set(4, cameraHeight)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return camera


def defineSoundTrackList():
    sounds = []
    music_dir = os.path.join(os.getcwd(), "music")
    for file in glob.glob(os.path.join(music_dir, "*.mp3")):
        sounds.append(os.path.basename(file))  # 파일명만 저장
    for file in glob.glob(os.path.join(music_dir, "*.wav")):
        sounds.append(os.path.basename(file))  # WAV 파일도 지원
    return sounds


def initializeKeyboard(soundPlayer):
    soundsList = defineSoundTrackList()
    defineWhiteNoteKeys(soundsList, soundPlayer)
    defineBlackNoteKeys(soundsList, soundPlayer)
    return buttonList


def defineWhiteNoteKeys(musicSoundsList, soundPlayer):
    for i in range(len(whiteNotes)):
        sound = defineSoundForSpecificKey(whiteNotes[i], musicSoundsList)
        if sound:
            buttonList.append(
                Button(whiteNotes[i], [i * widthWhiteNoteKey + shiftWhiteNote, int(heightWhiteNoteKey / 3)], whiteColor,
                       sound, [widthWhiteNoteKey, heightWhiteNoteKey]))


def defineBlackNoteKeys(musicSoundsList, soundPlayer):
    counter = 0
    tracer = 0
    checker = False

    for i in range(len(blackNotes)):
        tracer += 1
        sound = defineSoundForSpecificKey(blackNotes[i], musicSoundsList)
        if sound:
            buttonList.append(
                Button(blackNotes[i],
                       [shiftBlackNote + (i * int(1.5 * widthBlackNoteKey)) + (counter * int(1.3 * widthBlackNoteKey)),
                        int(0.5 * heightBlackNoteKey)], blackColor,
                       sound,
                       [widthBlackNoteKey, heightBlackNoteKey]))

        if tracer == 2 and checker is False:
            counter += 1
            tracer = 0
            checker = True

        if tracer == 3 and checker is True:
            counter += 1
            tracer = 0
            checker = False


def defineSoundForSpecificKey(buttonName, soundsList):
    possible_extensions = [".mp3", ".wav"]
    for ext in possible_extensions:
        sound_filename = f"{buttonName}{ext}"
        if sound_filename in soundsList:
            return sound_filename
    print(f"Not exist specific sound for {buttonName}")
    return None  # 소리가 없을 경우 None 반환


def showKeyboard(img, notesList):
    overlayImage = img.copy()

    for note in notesList:
        x, y = note.position
        width, height = note.size

        if note.color == whiteColor:
            cv2.rectangle(overlayImage, tuple(note.position), (x + width, y + height), (255, 255, 255), cv2.FILLED)
            cv2.rectangle(overlayImage, tuple(note.position), (x + width, y + height), (0, 0, 0), 2)

        if note.color == blackColor:
            cv2.rectangle(overlayImage, tuple(note.position), (x + width, y + height), (0, 0, 0), cv2.FILLED)
            cv2.rectangle(overlayImage, tuple(note.position), (x + width, y + height), (0, 0, 0), 2)

    alpha = 0.5  # Factor of transparency

    img = cv2.addWeighted(overlayImage, alpha, img, 1 - alpha, 0)

    return img


def checkBendFingers(landmarkList, img):
    bendTipsList = []
    pressedButton = []

    if len(landmarkList) != 0:
        for i in range(len(landmarkList)):

            rightThumbIsBent = landmarkList[i][tipsId[0] - 1][1] - landmarkList[i][tipsId[0]][1] < 10 and \
                                landmarkList[i][tipsId[0]][3] == "Right"

            leftThumbIsBent = landmarkList[i][tipsId[0]][1] - landmarkList[i][tipsId[0] - 1][1] < 10 and \
                               landmarkList[i][tipsId[0]][3] == "Left"

            if rightThumbIsBent:
                bendTipsList.append(landmarkList[i][tipsId[0]])
                # print("Right thumb was bent")

            if leftThumbIsBent:
                bendTipsList.append(landmarkList[i][tipsId[0]])
                # print("Left thumb was bent")

            for index in range(1, 5):
                fingerIsBent = landmarkList[i][tipsId[index] - 2][2] - landmarkList[i][tipsId[index]][2] <= 35
                if fingerIsBent:
                    bendTipsList.append(landmarkList[i][tipsId[index]])
                    # print("Finger was bent, id: " + str(id+1))

        pressedButton = checkIfButtonIsPressed(bendTipsList, img)

    return pressedButton, img


def checkIfButtonIsPressed(fingerBend, img):
    pressedButton = []

    if fingerBend:
        for button in buttonList:
            x, y = button.position
            width, height = button.size
            color = button.color

            for finger in fingerBend:
                fingerOverKeyXCoord = x < finger[1] < x + width
                if color == whiteColor:
                    fingerOverKeyYCoord = y + heightBlackNoteKey < finger[2] < y + height
                else:
                    fingerOverKeyYCoord = y < finger[2] < y + heightBlackNoteKey

                if fingerOverKeyXCoord and fingerOverKeyYCoord:
                    pressedButtonColor = generateRandomColor()
                    cv2.rectangle(img, tuple(button.position), (x + width, y + height), pressedButtonColor, cv2.FILLED)
                    pressedButton.append(button)
                    break  # 한 손가락당 하나의 키만 눌리도록

    return pressedButton


def generateRandomColor():
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    color = (blue, green, red)
    return color


def createMusicFrameToPlay(pressedButtonList):
    currentMusicFrame = []

    for button in pressedButtonList:
        if button.sound:
            currentMusicFrame.append(button.sound)

    return currentMusicFrame


def initializeSystem():
    camera = setCaptureDeviceSetting()
    soundPlayer = SoundPlayer()
    notes = initializeKeyboard(soundPlayer)
    detector = HandDetector()

    return camera, notes, detector, soundPlayer


def playBuildMusicForFrame(currentMusicFrameToPlay, previousMusicFrameToPlay, sound_player):
    if currentMusicFrameToPlay == previousMusicFrameToPlay:
        pass
    elif currentMusicFrameToPlay != previousMusicFrameToPlay:

        soundToTurnOff = list(set(previousMusicFrameToPlay) - set(currentMusicFrameToPlay))

        if soundToTurnOff:
            for sound in soundToTurnOff:
                sound_player.stop_sound(sound)

        soundToPlay = list(set(currentMusicFrameToPlay) - set(previousMusicFrameToPlay))

        for note in soundToPlay:
            sound_player.play_sound(note)


def main():
    captureDevice, notesList, detector, sound_player = initializeSystem()

    previousTime = 0
    previousMusicFrameToPlay = []

    while captureDevice.isOpened():
        success, img = captureDevice.read()
        if not success:
            print("Failed to read from camera.")
            break

        img = detector.detectHands(img)
        img = showKeyboard(img, notesList)
        landmark = detector.findLandmarkList(img)
        pressedButtonList, img = checkBendFingers(landmark, img)

        currentMusicFrameToPlay = createMusicFrameToPlay(pressedButtonList)
        playBuildMusicForFrame(currentMusicFrameToPlay, previousMusicFrameToPlay, sound_player)
        previousMusicFrameToPlay = currentMusicFrameToPlay

        currentTime = time.time()
        if previousTime != 0:
            fps = 1 / (currentTime - previousTime)
        else:
            fps = 0
        previousTime = currentTime

        cv2.putText(img, f"Fps: {int(fps)}", (cameraWidth - 150, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if not captureDevice.isOpened():
        print("Camera is not connected properly!")
        exit()

    captureDevice.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()
