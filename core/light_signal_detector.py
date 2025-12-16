import cv2
import numpy as np
from utils.drawing import draw_light_zone

class LightSignalDetector:
    def __init__(self, h, w, **kwargs):
        
        self.straight_light_zones = []  # List of polygons defining straight light zones
        self.left_light_zones = []      # List of polygons defining left turn light zones
        self.right_light_zones = []     # List of polygons defining right turn light zones

        self.draw_zones(kwargs.get('frame', None), kwargs.get('window_name', "Traffic Violation Detection"))

        self.zone_masks = {
            'straight': [],
            'left': [],
            'right': []
        }

        self.build_zone_mask(h, w)

    def build_zone_mask(self, h, w):
        def make_masks(zones):
            masks = []
            for polygon in zones:
                mask = np.zeros((h, w), dtype=np.uint8)
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
                masks.append(mask)
            return masks
        
        self.zone_masks['straight'] = make_masks(self.straight_light_zones)
        self.zone_masks['left'] = make_masks(self.left_light_zones)
        self.zone_masks['right'] = make_masks(self.right_light_zones)

    def detect_light_signals(self, image):
        """Detect light signals in the defined zones.

        Args:
            image (_type_): input image/frame

        Returns:
            _type_: return 3 lists of detected light signals for left, right, and straight directions. If a list is empty, it means no zones were defined for that direction.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red1 = cv2.inRange(hsv, (0, 20, 50), (30, 255, 255))
        red2 = cv2.inRange(hsv, (160, 20, 50), (180, 255, 255))

        red_mask = cv2.bitwise_or(red1, red2)

        yellow_mask = cv2.inRange(hsv, (20, 70, 50), (35, 255, 255))

        green_mask = cv2.inRange(hsv, (55, 70, 120), (95, 255, 255))

        candidates = {
            'left': None,
            'right': None,
            'straight': None
        }

        for direction in self.zone_masks:
            if len(self.zone_masks[direction]) == 0:
                continue
            scores = {'RED': 0, 'YELLOW': 0, 'GREEN': 0}
            counts = {'RED': 0, 'YELLOW': 0, 'GREEN': 0}
            for mask in self.zone_masks[direction]:
                r = cv2.countNonZero(cv2.bitwise_and(red_mask, red_mask, mask=mask))
                y = cv2.countNonZero(cv2.bitwise_and(yellow_mask, yellow_mask, mask=mask))
                g = cv2.countNonZero(cv2.bitwise_and(green_mask, green_mask, mask=mask))

                if r >= y and r >= g:
                    scores['RED'] += r; counts['RED'] += 1
                elif y >= g:
                    scores['YELLOW'] += y; counts['YELLOW'] += 1
                else:
                    scores['GREEN'] += g; counts['GREEN'] += 1

            best = max(scores, key=scores.get)
            if counts[best] == 0:
                score = 0
            else:
                score = scores[best] / counts[best]
            candidates[direction] = (best, score)

        return candidates['left'], candidates['straight'], candidates['right']

    
    def draw_zones(self, frame: np.ndarray, window_name="Traffic Violation Detection"):
        """Draw light signal zones interactively.

        Args:
            frame (np.ndarray): The video frame used as the canvas.
        """
        if frame is None:
            return
        
        categories = [
            ("Straight Light Signal Zones", self.straight_light_zones),
            ("Left Turn Light Signal Zones", self.left_light_zones),
            ("Right Turn Light Signal Zones", self.right_light_zones)
        ]

        for zone_name, zone_list in categories:
            points = draw_light_zone(frame, zone_name=zone_name, window_name=window_name)
            if len(points) >= 2:
                for i in range(0, len(points), 2):
                    top_left = points[i]
                    bottom_right = points[i + 1]
                    polygon = [
                        (top_left[0], top_left[1]),
                        (bottom_right[0], top_left[1]),
                        (bottom_right[0], bottom_right[1]),
                        (top_left[0], bottom_right[1])
                    ]
                    zone_list.append(polygon)