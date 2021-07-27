import tensorflow as tf
import cv2
import numpy as np
from random import choice
from datetime import datetime

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]


def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"


model = tf.keras.models.load_model('model.h5')

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_FPS, 25)

user_score = 0
comp_score = 0
prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle for user to play
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    # rectangle for computer to play
    cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)

    # extract the region of image within the user rectangle
    roi = frame[100:500, 100:500]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = np.array([img])
    
    
    # predict the move made
    pred = model.predict(img)
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)
        
    
    # predict the winner (human vs computer)
    if prev_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = choice(['rock', 'paper', 'scissors'])
            winner = calculate_winner(user_move_name, computer_move_name)
            if winner=="User":
                user_score+=1
            elif winner=="Computer":
                comp_score+=1
        else:
            computer_move_name = "none"
            winner = "Waiting..."
    prev_move = user_move_name

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Press 'q' to exit " ,(500, 690), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Your Move: " + user_move_name,
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "User Score: " + str(user_score),
                (125, 550), font, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer Score: " + str(comp_score),
                (825, 550), font, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
    if user_score>=10 and comp_score<10:
        winner = "User"
        cv2.putText(frame, "Winner: " + winner,(400, 650), font, 2, (0, 0, 255), 4, cv2.LINE_AA)
        user_score = 0
        comp_score = 0
    elif user_score<10 and comp_score>=10:
        winner = "Computer"
        cv2.putText(frame, "Winner: " + winner,(400, 650), font, 2, (0, 0, 255), 4, cv2.LINE_AA)
        user_score = 0
        comp_score = 0

    if computer_move_name != "none":
        icon = cv2.imread(
            "images//{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (400, 400))
        frame[100:500, 800:1200] = icon
    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
