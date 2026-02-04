import numpy as np

class PoseSequence:
    def __init__(self, sequence):
        self.poses = []
        for parts in sequence:
            self.poses.append(Pose(parts))
        
        # normalize poses based on the average torso pixel length
        torso_lengths = np.array([Part.dist(pose.neck, pose.lhip) for pose in self.poses if pose.neck.exists and pose.lhip.exists] +
                                 [Part.dist(pose.neck, pose.rhip) for pose in self.poses if pose.neck.exists and pose.rhip.exists])
        mean_torso = np.mean(torso_lengths)

        for pose in self.poses:
            for attr, part in pose:
                setattr(pose, attr, part / mean_torso)



class Pose:

    # Yolov8
    # 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow 8: Right Elbow 
    # 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle
    PART_NAMES = ['nose', 'leye', 'reye', 'lear', 'rear', 'lshoulder', 'rshoulder', 'lelbow', 'relbow', 'lwrist', 'rwrist', 'lhip', 'rhip', 'lknee', 'rknee', 'lankle', 'rankle']
    
    
    #  OpenPose
    # ['nose', 'neck',  'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle', 'reye', 'leye', 'rear', 'lear']
    # PART_NAMES = ['nose', 'neck',  'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle', 'reye', 'leye', 'rear', 'lear']

    def __init__(self, parts):
        """Construct a pose for one frame, given an array of parts

        Arguments:
            parts - 18 * 3 ndarray of x, y, confidence values
        """
        for name, vals in zip(self.PART_NAMES, parts):
            setattr(self, name, Part(vals))
    
    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value
    
    def __str__(self):
        out = ""
        for name in self.PART_NAMES:
            _ = "{}: {},{}".format(name, getattr(self, name).x, getattr(self, name).x)
            out = out + _ +"\n"
        return out
    
    def print(self, parts=None):
        if parts is None:
            parts = self.PART_NAMES
        out = ""
        for name in parts:
            if name not in self.PART_NAMES:
                raise NameError(name)
            _ = "{}: {},{}".format(name, getattr(self, name).x, getattr(self, name).y)
            out = out + _ + "\n"
        return out
    
    def get_part_xy(self, part=None):
        if part is None:
            return None
        else:
            return (getattr(self, part).x, getattr(self, part).y)

    def if_given_parts_exists(self, parts=None):
        if parts is None:
            return False
        else:
            return all(getattr(self, part).exists for part in parts)

class Part:
    def __init__(self, vals):
        self.x = vals[0]
        self.y = vals[1]
        self.c = vals[2]
        self.exists = self.c > 0.0

    def __floordiv__(self, scalar):
        __truediv__(self, scalar)

    def __truediv__(self, scalar):
        return Part([self.x / scalar, self.y / scalar, self.c])

    @staticmethod
    def dist(part1, part2):
        return np.sqrt(np.square(part1.x - part2.x) + np.square(part1.y - part2.y))