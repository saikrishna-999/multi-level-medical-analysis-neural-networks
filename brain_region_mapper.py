# brain_region_mapper.py

import cv2

# Brain region mapping: Dictionary linking region numbers to brain functions
BRAIN_REGION_MAP = {
    1: "Hippocampus - Memory and learning",
    2: "Cerebellum - Motor coordination",
    3: "Occipital Lobe - Visual processing",
    4: "Frontal Lobe - Decision making and behavior",
}

def map_region(cx, cy):
    """Identify brain region based on coordinates of the lesion."""
    if 50 <= cx <= 100 and 50 <= cy <= 100:
        return 1  # Hippocampus
    elif 150 <= cx <= 200 and 80 <= cy <= 120:
        return 2  # Cerebellum
    elif 200 <= cx <= 250 and 50 <= cy <= 100:
        return 3  # Occipital Lobe
    elif 100 <= cx <= 150 and 200 <= cy <= 250:
        return 4  # Frontal Lobe
    else:
        return None  # Unknown region

def highlight_differences(original_image_path, is_abnormal):
    """Highlight lesions and map affected brain regions."""
    original_image = cv2.imread(original_image_path)
    affected_regions = []

    if is_abnormal:
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(original_image, (cx, cy), 10, (0, 0, 255), -1)  # Red dot

                region = map_region(cx, cy)
                if region:
                    affected_regions.append(BRAIN_REGION_MAP[region])

    return original_image, affected_regions
