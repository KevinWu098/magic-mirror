import cv2
import numpy as np
from ultralytics import YOLO
import time # Import time module

# load yolo model
model = YOLO("yolo11x-seg.pt")
PERSON_CLASS_ID = 0 # Class ID for 'person' in COCO dataset

# set cam and res along with window name
WEBCAM_INDEX = 0
REQUESTED_WIDTH = 1920
REQUESTED_HEIGHT = 1080
WINDOW_NAME = "amogus sussy balls"

# init cam and make sure its working
cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Error: Could not open webcam index {WEBCAM_INDEX}.")
    exit()

# set res so we dont get the default 640x480 which is very cringe
cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)

# get actual res being used to make sure it worked
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Requested Resolution: {REQUESTED_WIDTH}x{REQUESTED_HEIGHT}")
print(f"Actual Resolution: {actual_width}x{actual_height}")

if actual_width != REQUESTED_WIDTH or actual_height != REQUESTED_HEIGHT:
     print("Warning: Could not set desired resolution. Check camera capabilities and drivers.")

# make resizable window thingy
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, actual_width, actual_height)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

prev_time = 0 # to calculate fps

while True:
    current_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    frame = cv2.flip(frame, 1) # mirror frame so its a mirror

    results = model(frame, classes=[PERSON_CLASS_ID], verbose=False) # krazy inference
    processed_frame = frame.copy()

    if results and results[0].masks is not None:
        result = results[0]
        masks = result.masks.data
        boxes = result.boxes

        for mask, box in zip(masks, boxes):
            class_id = int(box.cls.item())
            class_name = model.names[class_id]

            # generate color
            hue = int((class_id / len(model.names)) * 179)
            color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0].tolist()

            # process mask
            mask_np = mask.cpu().numpy()
            mask_np = cv2.resize(mask_np, (processed_frame.shape[1], processed_frame.shape[0]))
            mask_np = (mask_np > 0.5).astype(np.uint8)

            # create colored overlay
            overlay = np.zeros_like(processed_frame)
            overlay[mask_np == 1] = color

            # blend overlay with frame copy
            alpha = 0.3
            processed_frame = cv2.addWeighted(processed_frame, 1.0, overlay, alpha, 0)

            # draw bboxes and labels
            # x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            # label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            # cv2.putText(processed_frame, class_name, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    # draw fps text
    cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, processed_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
         current_prop = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
         cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL 
                               if current_prop == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)

cap.release()
cv2.destroyAllWindows()
    