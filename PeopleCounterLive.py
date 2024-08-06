# import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import datetime
with open('file.txt', 'a') as f:

  cap = cv2.VideoCapture(0)
  cap.set(3, 1280)
  cap.set(4, 720)

  model = YOLO("../Yolo-weights/yolov8n.pt")

  classNames = ["person ", " bicycle ", " car ", " motorbike ", " aeroplane ",
                " bus ", " train ", " truck ", " boat ", " traffic light ",
                " fire hydrant ", " stop sign ", " parking meter ", " bench ",
                " bird ", " cat ", " dog ", " horse ", " sheep ", " con ",
                " elephant ", " bear ", " zebra ", " giraffe ", " backpack ",
                " umbrella ", "handbag", "tie ", " suitcase ", " frisbee ",
                " skis ", " snowboard ", " sports ball ", " kite ", " baseball bat ",
                "baseball glove ", " skateboard ", " surfboard ", " tennis racket ",
                " bottle ", " wine glass ", " cup ", " fork ", " knife ", " spoon ",
                " bowl ", " banana ", " apple ", " sandwich ", " orange ", " broccoli ",
                " carrot ", " hot dog ", " pizza ", " donut ", " cake ", " chain ", " sofa ",
                " pottedplant ", " bed ", " diningtable ", " toilet ", " tvmonitor ", " Laptop ",
                " mouse ", " remote ", " keyboard ", " cell phone ", "microwave ", " oven ",
                " toaster ", " sink ", " refrigerator ", " book ", " clock ", " vase ",
                " scissors ", " teddy bear ", " hair drier ", " toothbrush "]
  mask = cv2.imread("mask.png")

  #tracking
  tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


  limits = [350,410,700,410]
  #limits2 = [150, 370, 390, 370]
  totalCountDown = []
  totalCount = []
  while True:
      success, img = cap.read()
      imgRegion = cv2.bitwise_and(img, mask)
      results = model(imgRegion, stream=True)

      detections= np.empty((0, 5))

      for r in results:
          boxes = r.boxes
          for box in boxes:
              x1, y1, x2, y2 = box.xyxy[0]
              x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

              w, h = x2-x1, y2-y1


              conf = math.ceil(box.conf[0]*100)/100
              #class name
              cls = int(box.cls[0])
              currentClass = classNames[cls]

              if currentClass == "person " or currentClass == " suitcase " or currentClass == " cell phone " and conf > 0.3:
                 currentArray = np.array([x1, y1, x2, y2, conf])
                 detections = np.vstack((detections, currentArray))


      resultsTracker = tracker.update(detections)

      cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
      #cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 0, 255), 5)

      now=datetime.datetime.now()
      for result in resultsTracker:
          x1,y1,x2,y2,Id = result
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
          #print(result)

          w, h = x2 - x1, y2 - y1
          cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2,colorR=(255,0,0))
          cvzone.putTextRect(img, f'{currentClass}{int(Id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)


          cx,cy = x1+w//2 , y1+h//2
          cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

          if limits[0]<cx< limits[2] and limits[1]-10<cy<limits[3]+10:
              if totalCount.count(Id)==0:
                  totalCount.append(Id)
                  cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

                  print(f' peopleCount: {len(totalCount)}', now.strftime("%d-%m-%y %H:%M:%S"), file=f)
                  print("  ", file=f)
          # if limits2[0] < cx < limits2[2] and limits2[1] - 10 < cy < limits2[1] + 10:
          #     if totalCountUp.count(Id) == 0:
          #         totalCountUp.append(Id)
          #         cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 255, 0), 5)
          #         print(f' countup: {len(totalCountUp)}', now.strftime("%d-%m-%y %H:%M:%S"), file=f)
          #         print(f' countdown: {len(totalCountDown)}', now.strftime("%d-%m-%y %H:%M:%S"), file=f)
          #         print("  ", file=f)


      cvzone.putTextRect(img, f' PeopleCount: {len(totalCount)}', (50, 50))
      #cvzone.putTextRect(img, f' countup: {len(totalCount)}', (50, 150))

      cv2.imshow("Image", img)
      cv2.imshow("imgRegion", imgRegion)
      cv2.waitKey(1)