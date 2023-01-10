import numpy as np
import matplotlib.pyplot as plt

MIN_ANGLE = 90
MAX_ANGLE = 80


class LinkMatrix():
    def __init__(self, length, theta, d=False) -> None:
        # ori
        self.mat_o = np.identity(3)
        self.mat_o[0][2] = -.036
        self.mat_o[1][2] = 0
        self.mat_o[2][2] = 1

        # translation
        self.mat_t = np.identity(3)
        self.mat_t[0][2] = length
        self.mat_t[1][2] = 0
        self.mat_t[2][2] = 1

        # theta
        self.mat_r = np.identity(3)
        self.mat_r[0][0] = np.cos(theta)
        self.mat_r[0][1] = -np.sin(theta)
        self.mat_r[1][0] = np.sin(theta)
        self.mat_r[1][1] = np.cos(theta)
        self.mat = self.mat_t @ self.mat_r
        self.vec = None


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


starting_hand = {"finger_lengths": [.072, .072, .072, .072],
                 "finger_matrices": [], "angle": MIN_ANGLE, "palm_width": .072}

l1_len = starting_hand["finger_lengths"][0]
l2_len = starting_hand["finger_lengths"][1]
palm_len = starting_hand["palm_width"]
palm_f1 = LinkMatrix(-palm_len / 2, 0)
#palm_f1.mat = np.linalg.inv(palm_f1.mat)
palm_f1.vec = np.array([-.036, 0, 1])
l1 = LinkMatrix(l1_len, .78)
theta = .78
mat_r = np.identity(3)
mat_r[0][0] = np.cos(theta)
mat_r[0][1] = -np.sin(theta)
mat_r[1][0] = np.sin(theta)
mat_r[1][1] = np.cos(theta)


#l1.mat = l1.mat @ np.linalg.inv(palm_f1.mat)
l1.mat = l1.mat @ mat_r
l2 = LinkMatrix(l2_len, .68)


l2.vec = (np.linalg.inv(l1.mat @ np.linalg.inv(palm_f1.mat)) @ np.linalg.inv(l2.mat) @ np.array([0, 0, 1]))
#l1.vec = palm_f1.mat @ np.linalg.inv(l1.mat) @ np.linalg.inv(palm_f1.mat) @ np.array([0, 0, 1])
l1.vec = (np.linalg.inv(l1.mat @ np.linalg.inv(palm_f1.mat)) @ np.array([0, 0, 1]))
palm_f1.vec = np.linalg.inv(palm_f1.mat) @ np.array([0, 0, 1])
f = plt.figure()
f.set_figwidth(10)
f.set_figheight(10)
plt.xlim(-.2, .2)
plt.ylim(-.2, .2)

# draw palm
x = [-palm_len/2, palm_len/2]
y = [0, 0]
total = np.sqrt(((x[1] - x[0])**2 + (y[1] - y[0])**2))
print(total)

plt.plot(x, y)


#l1.rotate(1.58, palm_f1.mat)
# draw l1
x = [x[0], l1.vec[0]]
y = [y[0], l1.vec[1]]
total = np.sqrt(((x[1] - x[0])**2 + (y[1] - y[0])**2))
print(total)

plt.plot(x, y)

#l2.rotate(0, l1.mat)
# draw l2
x = [l1.vec[0], l2.vec[0]]
y = [l1.vec[1], l2.vec[1]]
total = np.sqrt(((x[1] - x[0])**2 + (y[1] - y[0])**2))
print(total)

plt.plot(x, y)

l1_len = starting_hand["finger_lengths"][0]
l2_len = starting_hand["finger_lengths"][1]
palm_len = starting_hand["palm_width"]
palm_f1 = LinkMatrix(palm_len / 2, 0)
#palm_f1.mat = np.linalg.inv(palm_f1.mat)
palm_f1.vec = np.array([.036, 0, 1])
l1 = LinkMatrix(l1_len, 3.14-.68)
l1.mat = l1.mat @ np.linalg.inv(palm_f1.mat)
l2 = LinkMatrix(l2_len, -.68)


l2.vec = (np.linalg.inv(l1.mat) @ np.linalg.inv(l2.mat) @ np.array([0, 0, 1]))
#l1.vec = palm_f1.mat @ np.linalg.inv(l1.mat) @ np.linalg.inv(palm_f1.mat) @ np.array([0, 0, 1])
l1.vec = (np.linalg.inv(l1.mat) @ np.array([0, 0, 1]))
palm_f1.vec = np.linalg.inv(palm_f1.mat) @ np.array([0, 0, 1])

# draw palm
x = [-palm_len/2, palm_len/2]
y = [0, 0]


#l1.rotate(1.58, palm_f1.mat)
# draw l1
x = [x[1], l1.vec[0]]
y = [y[1], l1.vec[1]]
total = np.sqrt(((x[1] - x[0])**2 + (y[1] - y[0])**2))
print(total)

plt.plot(x, y)

#l2.rotate(0, l1.mat)
# draw l2
x = [l1.vec[0], l2.vec[0]]
y = [l1.vec[1], l2.vec[1]]
total = np.sqrt(((x[1] - x[0])**2 + (y[1] - y[0])**2))
print(total)
plt.plot(x, y)


# t = np.identity(3)
# t[0][2] = .005
# test = t @ l1.mat
# test = np.linalg.inv(test) @ np.array([0, 0, 1])
# x = [l1.vec[0], test[0]]
# y = [l1.vec[1], test[1]]
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
# plt.axis('square')
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
