import numpy as np
import cv2
import MCDWrapper
import time


np.set_printoptions(precision=8, suppress=True)
cap = cv2.VideoCapture('/home/sabra-pc/moving_object/PyFastMCD-master/t.mkv')
mcd = MCDWrapper.MCDWrapper()
isFirst = True


# used to record the time when we processed last frame
prev_frame_time = 0
  
# used to record the time at which we processed current frame
new_frame_time = 0

while(cap.isOpened()):
    ret, frame = cap.read()

    start_time = time.time()
    new_width = int(frame.shape[1]/3)
    new_height = int(frame.shape[0]/3)

    frame = cv2.resize(frame, (new_width, new_height))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape
    isResized = False

    if width%4 !=0 or height%4!=0:
        gray = cv2.resize(gray, ( 4 * (width//4), 4 * (height//4)))
        isResized = True


    mask = np.zeros(gray.shape, np.uint8)




    if (isFirst):
        mcd.init(gray)
        isFirst = False
    else:
        mask = mcd.run(gray)

    frame[mask > 0, 2] = 255


    #frame[mask == 255, 2] = 255
    #frame[mask == 100, 0] = 255


    if isResized:
        mask = cv2.resize(mask, (width, height))



    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    end_time = time.time()

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(end_time-start_time)

    # converting the fps into integer
    fps = int(fps)
  
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
  
    # putting the FPS count on the frame
    cv2.putText(frame, fps, (20, 70), font, 2, (100, 255, 0), 3, cv2.LINE_AA)


    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


