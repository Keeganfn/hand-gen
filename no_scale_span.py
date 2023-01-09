import numpy as np
import matplotlib.pyplot as plt

MIN_ANGLE = 90
MAX_ANGLE = 80


class LinkMatrix():
    def __init__(self, length, theta) -> None:
        mat_o = np.identity(3)
        mat_o[0][2] = -.032 / 2
        mat_o[1][2] = 0
        mat_o[2][2] = 1

        # translation
        mat_t = np.identity(3)
        mat_t[0][2] = length
        mat_t[1][2] = 0
        mat_t[2][2] = 1

        # theta
        mat_r = np.identity(3)
        mat_r[0][0] = np.cos(theta)
        mat_r[0][1] = -np.sin(theta)
        mat_r[1][0] = np.sin(theta)
        mat_r[1][1] = np.cos(theta)
        #mat_t = mat_r @ mat_t

        print(mat_r)
        print(mat_t)
        # mat_test = np.identity(3)
        # mat_test[0][2] = -.032 / 2
        # mat_test[0][0] = np.cos(theta)
        # mat_test[0][1] = -np.sin(theta)
        # mat_test[1][0] = np.sin(theta)
        # mat_test[1][1] = np.cos(theta)
        #self.mat = mat_r @ mat_t
        self.mat = mat_t @ mat_r
        #self.mat = mat_r @ mat_t @ mat_o

        self.vec = np.linalg.inv(self.mat) @ np.array([0, 0, 1])
        #self.vec = mat_t @ mat_r @ np.array([0, 0, 1])
        #self.vec = self.vec @ mat_r
        #self.vec = mat_t @ self.mat
        #self.vec = self.vec @ np.array([0, 0, 1])


palm_w = 0
l1 = 0
l2 = 0
l3 = 0
l4 = 0
a1 = 0
a2 = 0
cube_dims = [.039, .039]

hand_ratio = {"J1toJ2": 0, "J3toJ4": 0}
hand_desc = {"finger_lengths": [], "angle": 0, "palm_width": 0}


T_J1 = np.ones((3, 3))
T_J1[1]


starting_hand = {"finger_lengths": [.032, .072, .072, .072],
                 "finger_matrices": [], "angle": MIN_ANGLE, "palm_width": .072}

l1_len = starting_hand["finger_lengths"][0]
angle = starting_hand["angle"]
l1 = LinkMatrix(l1_len, 1.57)
print(l1.mat)
print(l1.vec)
plt.xlim(-.05, .05)
plt.ylim(-.05, .05)
x = [-.032 / 2, .032 / 2]
y = [0, 0]
plt.plot(x, y)
x = [0, l1.vec[0]]
y = [0, l1.vec[1]]
plt.plot(x, y)
t = np.identity(3)
t[0][2] = .005
test = t @ l1.mat
test = np.linalg.inv(test) @ np.array([0, 0, 1])
x = [l1.vec[0], test[0]]
y = [l1.vec[1], test[1]]
plt.plot(x, y)
plt.show()


def draw():
    x = [-.032 / 2, .032 / 2]
    y = [0, 0]
    plt.plot(x, y)
    x = [-.032 / 2, -.032 / 2]
    y = [0, .05]
    plt.plot(x, y)
    plt.show()


# draw()
