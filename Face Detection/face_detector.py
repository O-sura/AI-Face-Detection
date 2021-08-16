import cv2

train_dataset = cv2.CascadeClassifier("D:/gAMes/VS projects/Face Detection/haarcascade_frontalface_default.xml")

#face = cv2.imread("E:/sudu/IMG-20200830-WA0019.jpg")
#face = cv2.imread("E:/download_20201209_210713.jpg")

#face = cv2.imread("D:/gAMes/VS projects/Face Detection/girl.png")

webcam = cv2.VideoCapture(0)

while True:

    sucessful_frame_read, frame = webcam.read()
    greyscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = train_dataset.detectMultiScale(greyscale_img)


    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)


    cv2.imshow("Face Detector App", frame)
    key = cv2.waitKey(1)

    if key == 85 or key == 113:
        break

webcam.release()





print("Code Executed! Good Job")