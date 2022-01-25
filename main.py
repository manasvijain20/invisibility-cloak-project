import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

#capturing the video using webcam
cap = cv2.VideoCapture(0)
time.sleep(2)
bg = 0

for i in range(60):
    ret,bg = cap.read()

bg = np.flip(bg, axis=1)

while(cap.isOpened()):
    ret,img = cap.read()
    if not ret:
        break
    img = np.flip(img, axis=1)
    #as we capture the frame we are also capturing the colors hence we will convert bgr(blue, black, green) to hsv(hue, saturation, value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #generating mask to detect black color
    lower_black = np.array([0,120,50])
    upper_black = np.array([10,255,255])
    mask_1 = cv2.inRange(hsv, lower_black, upper_black)
    lower_black = np.array([30,30,0])
    upper_black = np.array([104,153,70])
    mask_2 = cv2.inRange(hsv, lower_black, upper_black)
    mask_1 = mask_1+mask_2
    #open and expand the image where there is mask 1 with color
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))
    #sleecting only the part that does not have mask and saving it in mask 2
    mask_2 = cv2.bitwise_not(mask_1)
    #keeping only the part of the images without the black color
    res_1 = cv2.bitwise_and(img,img,mask=mask_2)
    #keeping only the part of images with black color
    res_2 = cv2.bitwise_and(bg,bg,mask = mask_1)
    #generating only the part of images with black color
    final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)
    output_file.write(final_output)
    #displaying the output of the user
    cv2.imshow("magic", final_output)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
