import math
import mediapipe

class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    # Функция перевода Landmark, библиотеки Mediapipe, в векторный формат
    def landmark_to_vector(landmark: 'mediapipe.framework.formats.landmark_pb2.NormalizedLandmark'):
        return Vector(landmark.x, landmark.y, landmark.z)

    def __str__(self):
        return "Vector class\nx: " + str(self.x) + "\ny: " + str(self.y) + "\nz: " + str(self.z) + "\n"

    def __add__(self, other: 'Vector'):
        return Vector(self.x + other.x,
                      self.y + other.y,
                      self.z + other.z)

    def __iadd__(self, other: 'Vector'):
        self.x += other.x
        self.y += other.y
        self.z += other.z

    def __sub__(self, other: 'Vector'):
        return Vector(self.x - other.x,
                      self.y - other.y,
                      self.z - other.z)

    def __isub__(self, other: 'Vector'):
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z

    def __truediv__(self, other):
        return Vector(self.x / other,
                      self.y / other,
                      self.z / other)

    def get_xy_angle(self):
        angle = math.acos(-self.y / math.sqrt(pow(self.x, 2) + pow(self.y, 2)))
        if (self.x > 0):
            angle = -angle
        return angle

    def length(self):
        return math.sqrt(pow(self.x, 2) + pow(self.y, 2) + pow(self.z, 2))

    def rotate_y(self, angle: float):
        len = math.sqrt(pow(self.x, 2) + pow(self.y, 2))
        ang = math.acos(-self.y / math.sqrt(pow(self.x, 2) + pow(self.y, 2)))
        if self.x < 0:
            ang = -ang
        ang += angle
        self.x = math.sin(ang) * len
        self.y = -math.cos(ang) * len

    def reverse(self):
        len = math.sqrt(pow(self.x, 2) + pow(self.y, 2))
        ang = math.acos(-self.y / math.sqrt(pow(self.x, 2) + pow(self.y, 2)))
        if self.x < 0:
            ang = -ang
        ang = -ang
        self.x = math.sin(ang) * len
        self.y = -math.cos(ang) * len

    # def rotate_z(self, angle):

