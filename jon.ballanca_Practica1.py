import math
import numpy as np
from numpy import dtype
from numpy.ma.core import transpose


np.set_printoptions(precision = 1, suppress = True)

def calculate_g(number):
    numerator = 1
    denominator = 1 + math.e**(-number)

    return numerator/denominator

def feedForward(wUno, x_column, b1, wDos, b2):
    zUno = np.array([[0],
                     [0],
                     [0]], dtype=float)

    aUno = np.array([[0],
                     [0],
                     [0]], dtype=float)

    zDos = np.array([[0],
                     [0]], dtype=float)

    aDos = np.array([[0],
                     [0]], dtype=float)

    for i in range(len(wUno)):
        zUno[i] = np.dot(wUno[i], x_column) + b1[i]
        aUno[i] = calculate_g(zUno[i])

    for i in range(len(wDos)):
        zDos[i] = np.dot(wDos[i], aUno) + b2[i]
        aDos[i] = calculate_g(zDos[i])

    return aDos, zDos, zUno, aUno

# Inputs:
factor = np.sqrt(1 / 2)

w1 = np.random.randn() * factor
w2 = np.random.randn() * factor
w3 = np.random.randn() * factor
w4 = np.random.randn() * factor
w5 = np.random.randn() * factor
w6 = np.random.randn() * factor
w7 = np.random.randn() * factor
w8 = np.random.randn() * factor
w9 = np.random.randn() * factor
w10 = np.random.randn() * factor
w11 = np.random.randn() * factor
w12 = np.random.randn() * factor

wUno = np.array([[w1, w4],
                 [w2, w5],
                 [w3, w6]], dtype=float)

wDos = np.array([[w7, w9, w11],
                 [w8, w10, w12]], dtype=float)

b1 = np.array([[np.random.randn()], [np.random.randn()], [np.random.randn()]], dtype=float)
b2 = np.array([[np.random.randn()], [np.random.randn()]], dtype=float)


x1 = np.random.randn()
x2 = np.random.randn()


x_column = np.array([[x1],
                     [x2]], dtype=float)

# Outputs
zUno = np.array([[0],
                 [0],
                 [0]], dtype=float)

aUno = np.array([[0],
                 [0],
                 [0]], dtype=float)

zDos = np.array([[0],
                 [0]], dtype=float)

aDos = np.array([[0],
                 [0]], dtype=float)



# Initial calculation of aDos

aDos, zDos, zUno, aUno = feedForward(wUno, x_column, b1, wDos, b2)


# Backwards Propagation
alpha = 0.25

# Layer 2

xin = np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])

output = np.array([[0, 0],
                    [1, 0],
                    [1, 0],
                    [0, 1]])

e = 0

for count in range(5000):

    if e == 4:
        e = 0

    y = np.array([[output[e][0]], [output[e][1]]])
    x_column = np.array([[xin[e][0]], [xin[e][1]]])

    ###### feedForward Repetition
    aDos, zDos, zUno, aUno = feedForward(wUno, x_column, b1, wDos, b2)
    ######

    gradient_zDos = aDos - y

    gradient_wDos = np.dot(gradient_zDos, np.transpose(aUno))
    gradient_b2 = gradient_zDos

    wDos = wDos - (alpha * gradient_wDos)
    b2 = b2 - (alpha * gradient_b2)

# Layer 1
    gradient_zUno = np.multiply(np.dot((np.transpose(wDos)), gradient_zDos), (aUno * (1 - aUno)))
    gradient_wUno = np.dot(gradient_zUno, np.transpose(x_column))
    gradient_b1 = gradient_zUno

    wUno = wUno - (alpha * gradient_wUno)
    b1 = b1 - (alpha * gradient_b1)

    e += 1

print("")

print("Input   XOR AND")

for i in range(4):
    y = np.array([[output[i][0]], [output[i][1]]])
    x_column = np.array([[xin[i][0]], [xin[i][1]]])

    aDosAux, zDosAux, zUnoAux, aUnoAux = feedForward(wUno, x_column, b1, wDos, b2)
    print(x_column.flatten(), " ", aDosAux.flatten())