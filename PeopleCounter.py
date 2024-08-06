from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import datetime

# import pyodbc
# import face_recognition


# conn_str = (
#     r'DRIVER={SQL Server};'
#     r'SERVER=INBOOK_X1_PRO;'
#     r'DATABASE=People counter;'
# )


# conn = pyodbc.connect(conn_str)


cap = cv2.VideoCapture("C:\\Projects Raya\\people counter\\Videos\\test111.mp4")

model = YOLO("../Yolo-weights-v8/yolov8n.pt")

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
              " pottedplant ", " bed ", "diningtable ", " toilet ", " tvmonitor ", " Laptop ",
              " mouse ", " remote ", " keyboard ", " cell phone ", "microwave ", " oven ",
              " toaster ", " sink ", " refrigerator ", " book ", " clock ", " vase ",
              " scissors ", " teddy bear ", " hair drier ", " toothbrush "]
mask3 = cv2.imread("C:\\Projects Raya\\people counter\\people counter\\mask3.png")

# tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [530, 330, 770, 330]
limits2 = [530, 300, 770, 300]
totalCountOut = []
totalCountIn = []

now = datetime.datetime.now()
print(f' start time:', now.strftime("%d-%m-%y %H:%M:%S"))
# cursor = conn.cursor()
while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask3)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            w, h = x2 - x1, y2 - y1

            conf = math.ceil(box.conf[0] * 100) / 100
            # class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person " or currentClass == " cell phone " and conf > 0.3:
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{currentClass}{int(Id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 10 < cy < limits[3] + 10:
            if totalCountOut.count(Id) == 0 and totalCountIn.count(Id) == 0:
                totalCountOut.append(Id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

                print(f' Exit: {len(totalCountOut)}', now.strftime("%d-%m-%y %H:%M:%S"))
                # cursor.execute("INSERT INTO Transactions (typeID,transactionDate) VALUES (?,?)", ("2", now))
                # conn.commit()

        if limits2[0] < cx < limits2[2] and limits2[1] - 10 < cy < limits2[1] + 10:
            if totalCountIn.count(Id) == 0 and totalCountOut.count(Id) == 0:
                totalCountIn.append(Id)

                cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 255, 0), 5)

                print(f' Enter: {len(totalCountIn)}', now.strftime("%d-%m-%y %H:%M:%S"))
                # cursor.execute("INSERT INTO Transactions (typeID,transactionDate) VALUES (?,?)", ("1", now))
                # conn.commit()

    cvzone.putTextRect(img, f' Enter: {len(totalCountIn)}', (50, 50))
    cvzone.putTextRect(img, f' Exit: {len(totalCountOut)}', (50, 150))

    cv2.imshow("Counter", img)
    cv2.waitKey(1)

conn.close()

