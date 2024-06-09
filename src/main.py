import functools
import math
import time

import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt, animation


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def RGBtoGrayScale(image):
    result = image.copy()
    result = (result[:, :, 0] + result[:, :, 1] + result[:, :, 2]) / 3

    result = result.reshape((image.shape[0], image.shape[1]))

    return result


def mapOT(image, mapping):
    return image.reshape((-1, 3))[mapping].reshape(image.shape)


def toCycle(perm):
    pi = {i: perm[i] for i in range(len(perm))}
    cycles = []
    while pi:
        elem0 = next(iter(pi))
        this_element = pi[elem0]  # already the next one?
        next_item = pi[this_element]

        cycle = []

        while True:
            cycle.append(this_element)
            del pi[this_element]
            this_element = next_item
            if next_item in pi:
                next_item = pi[next_item]
            else:
                break
        cycles.append(cycle)

    return cycles


def permPower(perm: np.ndarray, power: int):
    result = np.arange(len(perm))
    intermediate_power = perm
    while power != 0:
        if power % 2 == 1:
            result = result[intermediate_power]
        intermediate_power = intermediate_power[intermediate_power]
        power = power >> 1
    return result


def permLength(perm: np.ndarray):
    perm_lengths = list(map(len, toCycle(perm)))
    print("perm_lengths", sorted(perm_lengths))
    return math.lcm(*perm_lengths)


def lerpImg(start_img: np.ndarray, target_img: np.ndarray, x: float):
    dx = target_img-start_img
    return start_img+(dx*x)
    #.astype(np.int32)

def mongeMap(source: np.ndarray, target: np.ndarray):
    sorted_perm_target = np.argsort(target.flatten())
    sorted_perm_source = np.argsort(source.flatten())

    pixel_count = len(source.flatten())
    if pixel_count != len(target.flatten()):
        print("amount of pixels does not match, cannot construct monge map")
        exit(0)

    target_inverse = np.ndarray((pixel_count,), dtype=np.int32)  # 16 is clearly not enough

    target_inverse[sorted_perm_target] = list(range(pixel_count))
    return sorted_perm_source[target_inverse]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # constraint: images must be of equal dimensions

    imgTarget = mpimg.imread("../res/hydriven H.png")
    imgSource = mpimg.imread("../res/GTT H.png")

    OTmap = mongeMap(RGBtoGrayScale(imgSource), RGBtoGrayScale(imgTarget))
    # print(OTmap.index(0))
    OTpermLength = permLength(OTmap)
    print("permutation degree", OTpermLength)

    fig, ax = plt.subplots()
    img = ax.imshow(mapOT(imgSource, OTmap))
    frame_text = ax.text(0.05, 0.95, '-1', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

    cyclelength = 465
    fps = 180
    def plotUpdate(frame, ot_map):
        # determine power based on frame
        # update the imshow artist with AxesImage.set_data
        perm_map = []
        power = frame

        if frame<fps:
            power = 0
        else:
            power = int((frame + 1-fps) * (2610794299037414490 / cyclelength))
        # print(frame, power)
        new_img = imgTarget
        if frame >= cyclelength+3*fps:
            new_img = imgTarget
        elif frame >=cyclelength+2*fps:
            new_img = lerpImg(mapOT(imgSource, ot_map), imgTarget, min(0.005*(frame-cyclelength-2*fps), 1.0))
        elif frame >=cyclelength+1*fps:
            new_img = mapOT(imgSource, ot_map)
        else:
            perm_map = permPower(ot_map, power)
            new_img = mapOT(imgSource, perm_map)
        img.set_data(new_img)
        frame_text.set_text('%d' % frame)
        # must be iterable
        return [img, frame_text]


    ani = animation.FuncAnimation(fig=fig, func=(functools.partial(plotUpdate, ot_map=OTmap)), frames=cyclelength + 4*fps+2,
                                  interval=1,
                                  repeat_delay=1000)
    # plt.show()

    writervideo = animation.FFMpegWriter(fps=fps)
    start = time.time()
    ani.save('degrading.mp4', writer=writervideo)
    end = time.time()
    print("time spend", end - start)
    ani.save('degrading.mp4', writer=writervideo)
    plt.close()
