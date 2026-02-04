import numpy as np

# ================================
# Utility math functions
# ================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh2xyxy_vectorized(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]"""
    y = np.empty_like(x)
    xy = x[:, :2]  # center x, y
    wh = x[:, 2:] / 2  # half width, height
    y[:, :2] = xy - wh  # x1, y1
    y[:, 2:] = xy + wh  # x2, y2
    return y

def xywh2xyxy(box):
    x, y, w, h = box
    return np.array([
        x - w/2,
        y - h/2,
        x + w/2,
        y + h/2
    ])

def nms(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return keep

def get_center_of_bbox(bbox):
    """
    Reused from:
    https://github.com/abdullahtarek/football_analysis
    Original Author: Abdullah Tarek
    License: MIT    
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_wdth(bbox):
    """
    Reused from:
    https://github.com/abdullahtarek/football_analysis
    Original Author: Abdullah Tarek
    License: MIT    
    """
    return bbox[2] - bbox[0]

def measure_distance(p1, p2):
    """
    Reused from:
    https://github.com/abdullahtarek/football_analysis
    Original Author: Abdullah Tarek
    License: MIT    
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def measure_xy_distance(p1, p2):
    """
    Reused from:
    https://github.com/abdullahtarek/football_analysis
    Original Author: Abdullah Tarek
    License: MIT    
    """
    return ((p1[0] - p2[0], p1[1] - p2[1])) 

def get_foot_position(bbox):
    """
    Reused from:
    https://github.com/abdullahtarek/football_analysis
    Original Author: Abdullah Tarek
    License: MIT    
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)

def compute_performance_measures(time_stamps, start_time=None, end_time=None):
    
    # Validate start/end times
    if start_time is not None and end_time is not None:
        try:
            # Attempt to ensure they are numbers (int or float)
            s_time = float(start_time)
            e_time = float(end_time)

            if s_time > e_time:
                print(f"Warning: Start time ({s_time}) is after end time ({e_time}).")
            else:
                print(f"Total Execution Time: {(e_time - s_time) / 1e9:.4f} seconds")

        except ValueError:
            raise(f"Error: Time inputs must be numeric. Received: {type(start_time).__name__} and {type(end_time).__name__}")
        except TypeError:
            raise("Error: Invalid input types for time calculation.")

    # Compute latency
    # Ensure time_stamps contains numbers
    try:
        time_differences = [float(j) - float(i) for i, j in zip(time_stamps[:-1], time_stamps[1:])]
    except (ValueError, TypeError):
        raise "Error: Timestamp list contains non-numeric data."

    if time_differences:
        latency_ms = sum(time_differences) / len(time_differences)
        fps = 1000 / latency_ms if latency_ms > 0 else 0.0
    else:
        latency_ms = 0.0
        fps = 0.0

    print(f"Total frames processed: {len(time_stamps)}")
    print(f"Average FPS: {fps:.1f}")
    print(f"Per-frame latency: {latency_ms:.2f} ms")


def correct_boxes_for_camera(results, camera_movement):
    x_movement, y_movement = camera_movement
    results['boxes'] -= np.array([x_movement, y_movement, x_movement, y_movement])
    return results