import cv2
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
import math
import time

# Constants for gesture detection
DRAG_THRESHOLD = 30      # Distance threshold for drag detection
FINGER_CLOSED_THRESHOLD = 0.2  # Threshold for detecting closed fingers (for fist detection)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set up MediaPipe Hands
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2) as hands:
    
    # Variables to track hand gestures
    prev_positions = {}
    drag_start_position = None
    is_grabbing = False
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        # To improve performance, optionally mark the image as not writeable
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        # Draw the hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        current_time = time.time()
        
        # Check if hand landmarks are detected
        if hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
            multi_handedness = getattr(results, 'multi_handedness', [])
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand skeleton
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Get hand ID (left or right hand)
                handedness = "Unknown"
                if idx < len(multi_handedness) and hasattr(multi_handedness[idx], 'classification'):
                    if multi_handedness[idx].classification:
                        handedness = multi_handedness[idx].classification[0].label
                hand_id = f"{handedness}_{idx}"
                
                # Extract key points for all finger tips and palm
                index_tip = (hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y)
                middle_tip = (hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y)
                ring_tip = (hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y)
                pinky_tip = (hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y)
                thumb_tip = (hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y)
                palm_center = (hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y)  # Middle finger MCP as palm center
                
                # Get positions of finger bases
                index_mcp = (hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y)
                middle_mcp = (hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y)
                ring_mcp = (hand_landmarks.landmark[13].x, hand_landmarks.landmark[13].y)
                pinky_mcp = (hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y)
                
                # Convert normalized coordinates to pixel coordinates
                h, w, _ = image.shape
                palm_px = (int(palm_center[0] * w), int(palm_center[1] * h))
                
                # Check if fingers are extended
                index_extended = hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y - 0.04
                middle_extended = hand_landmarks.landmark[12].y < hand_landmarks.landmark[9].y - 0.04
                ring_extended = hand_landmarks.landmark[16].y < hand_landmarks.landmark[13].y - 0.04
                pinky_extended = hand_landmarks.landmark[20].y < hand_landmarks.landmark[17].y - 0.04
                
                # Check for fist gesture (fingers not extended)
                is_fist = (not index_extended and 
                          not middle_extended and 
                          not ring_extended and 
                          not pinky_extended)
                
                # Check for grab/fist gesture
                if is_fist:
                    if not is_grabbing:
                        is_grabbing = True
                        drag_start_position = palm_px
                        print(f"Grab detected at {palm_px}")
                    else:
                        # If already grabbing, calculate drag distance
                        if drag_start_position:
                            drag_distance = math.sqrt((palm_px[0] - drag_start_position[0])**2 + (palm_px[1] - drag_start_position[1])**2)
                            if drag_distance > DRAG_THRESHOLD:
                                drag_vector = (palm_px[0] - drag_start_position[0], palm_px[1] - drag_start_position[1])
                                print(f"Dragging: distance={drag_distance:.1f}px, vector={drag_vector}")
                                # Update drag start position for continuous tracking
                                drag_start_position = palm_px
                    # Visualize grabbing
                    cv2.circle(image, palm_px, 20, (0, 255, 0), -1)
                    cv2.putText(image, "GRAB", (palm_px[0] - 40, palm_px[1] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    is_grabbing = False
                    drag_start_position = None
                
                # Store current palm position for next frame
                prev_positions[hand_id] = {
                    "palm": palm_px,
                    "time": current_time
                }
        
        # Flip the image horizontally for a selfie-view display
        flipped_image = cv2.flip(image, 1)
        
        # Display info on screen
        cv2.putText(flipped_image, "Gesture: Fist to Grab & Drag", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the image
        cv2.imshow('Hand Tracking - Grab Detection', flipped_image)
        
        # Exit if ESC key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
