import cv2
import numpy as np
from ultralytics import YOLO
import time # Import time module

# load yolo model
model = YOLO("yolo11x-seg.pt")
PERSON_CLASS_ID = 0 # Class ID for 'person' in COCO dataset

def find_best_person(boxes, frame_width, frame_height):
    """
    Find the person closest to camera and most centered in the frame.
    
    Args:
        boxes: YOLO detection boxes
        frame_width: Width of the frame
        frame_height: Height of the frame
        
    Returns:
        Index of the best person or None if no people detected
    """
    if len(boxes) == 0:
        return None
        
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    
    best_score = -float('inf')
    best_idx = None
    
    for i, box in enumerate(boxes):
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Calculate box center
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        
        # Calculate distance from center (normalized by frame dimensions)
        center_dist = np.sqrt(
            ((box_center_x - frame_center_x) / frame_width) ** 2 + 
            ((box_center_y - frame_center_y) / frame_height) ** 2
        )
        
        # Calculate box size (area)
        box_size = (x2 - x1) * (y2 - y1)
        
        # Score combines size (closer = bigger) and centrality
        # Higher weight on size and centrality
        size_score = box_size / (frame_width * frame_height)  # Normalize by frame size
        center_score = 1 - center_dist  # Invert so center=1, edge=0
        
        score = 0.7 * size_score + 0.3 * center_score
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    return best_idx

def refine_mask(mask, kernel_size=5):
    """
    Apply morphological operations to refine the mask.
    
    Args:
        mask: Input binary mask
        kernel_size: Size of the morphological kernel
        
    Returns:
        Refined mask
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Close the mask to fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Dilate the mask slightly to extend the edges
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask

def get_segmentation_data(frame, retina_masks=True):
    """
    Get segmentation data for people in the frame, with the best person as the first element.
    
    Args:
        frame: Input image frame
        retina_masks: Whether to use high-quality retina masks for better segmentation
    
    Returns:
        List of dictionaries containing segmentation data, with best person first.
        Each dict contains: 
            - 'mask': numpy array of the mask
            - 'box': bounding box coordinates
            - 'class_id': class ID
            - 'class_name': class name
            - 'is_best': boolean indicating if this is the best person
    """
    # Use a lower confidence threshold to capture more of the body
    results = model(frame, classes=[PERSON_CLASS_ID], verbose=False, retina_masks=retina_masks, conf=0.2)
    segmentation_data = []
    
    if results and results[0].masks is not None:
        result = results[0]
        masks = result.masks.data
        boxes = result.boxes
        
        # Find the best person to segment
        best_idx = find_best_person(boxes, frame.shape[1], frame.shape[0])
        
        # Process all persons
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            
            # Process mask
            mask_np = mask.cpu().numpy()
            
            # Only resize if not using retina_masks (retina masks are already properly sized)
            if not retina_masks:
                mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
            
            mask_np = (mask_np > 0.5).astype(np.uint8)
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Extend the bottom of the mask a bit to compensate for possible cutoffs
            # Calculate the extension as a percentage of the body height
            extension_ratio = 0.05  # Extend by 5% of the body height
            bottom_extension = int((y2 - y1) * extension_ratio)
            
            # Create extended mask
            extended_mask = mask_np.copy()
            
            # Only extend for the bottom part and within image boundaries
            max_y = min(y2 + bottom_extension, frame.shape[0] - 1)
            
            # For pixels in the extension area, check if they're close to the bottom of the current mask
            if y2 < max_y:
                for y in range(y2, max_y + 1):
                    for x in range(max(x1, 0), min(x2, frame.shape[1])):
                        # Check if there's mask content in the row just above
                        if y > 0 and np.any(mask_np[y-3:y, max(x-2, 0):min(x+3, frame.shape[1]-1)] == 1):
                            extended_mask[y, x] = 1
            
            # Apply morphological operations to refine the mask
            refined_mask = refine_mask(extended_mask, kernel_size=5)
            
            # Store segmentation data
            data = {
                'mask': refined_mask,
                'box': box.xyxy[0].tolist(),
                'class_id': class_id,
                'class_name': class_name,
                'is_best': (i == best_idx)
            }
            
            # Add to list - will be sorted later
            segmentation_data.append(data)
    
    # Sort the segmentation data to put the best person first
    segmentation_data.sort(key=lambda x: not x['is_best'])
    
    return segmentation_data

if __name__ == "__main__":
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
    
    # Use retina masks by default for higher quality segmentation
    use_retina_masks = True
    
    # Display controls
    print("Controls:")
    print("  q: Quit")
    print("  f: Toggle fullscreen")
    print("  r: Toggle retina masks")
    
    while True:
        current_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        frame = cv2.flip(frame, 1) # mirror frame so its a mirror
    
        segmentation_data = get_segmentation_data(frame, retina_masks=use_retina_masks)
        processed_frame = frame.copy()
    
        # Visualize the segmentation data
        for data in segmentation_data:
            # Choose color - dynamic color for best person, blue for others
            if data['is_best']:
                # Generate dynamic color for the chosen person
                hue = int((data['class_id'] / len(model.names)) * 179)
                color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0].tolist()
            else:
                # Blue color for non-chosen people
                color = [255, 0, 0]  # BGR format - blue
    
            mask_np = data['mask']
            
            # Create colored overlay
            overlay = np.zeros_like(processed_frame)
            overlay[mask_np == 1] = color
    
            # Blend overlay with frame copy
            alpha = 0.3
            processed_frame = cv2.addWeighted(processed_frame, 1.0, overlay, alpha, 0)
    
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
    
        # draw fps text
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add text showing if retina masks are enabled
        mask_text = "Retina Masks: ON" if use_retina_masks else "Retina Masks: OFF"
        cv2.putText(processed_frame, mask_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        cv2.imshow(WINDOW_NAME, processed_frame)
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
             current_prop = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
             cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL 
                                   if current_prop == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)
        elif key == ord('r'):
            # Toggle retina masks on/off
            use_retina_masks = not use_retina_masks
            print(f"Retina masks {'enabled' if use_retina_masks else 'disabled'}")
    
    cap.release()
    cv2.destroyAllWindows()
    