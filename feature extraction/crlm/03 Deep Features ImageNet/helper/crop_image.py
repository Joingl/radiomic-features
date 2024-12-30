import pydicom
import numpy
from PIL import Image
def getHeightOfMask(mask):
    ones = mask.nonzero()
    height = ones[0].max() - ones[0].min()

    return height

def getWidthOfMask(mask):
    ones = mask.nonzero()
    width = ones[1].max() - ones[1].min(0)

    return width

def getCenterOfMask(mask):
    ones = mask.nonzero()
    height = getHeightOfMask(mask)
    width = getWidthOfMask(mask)

    y = ones[0].min() + height//2
    x = ones[1].min() + width//2

    return y, x

def cropImageTo96x96(img, mask):
    center = getCenterOfMask(mask)

    if center[1] >= 48:
        top_left = (center[0]-48, center[1]-48)  # (height, width)
        bot_right = (center[0]+48, center[1]+48)  # (height, width)

    elif center[1] < 48:
        top_left = (center[0]-48, center[1]-38)  # (height, width)
        bot_right = (center[0]+48, center[1]+58)  # (height, width)

    cropped_img = img[top_left[0]:bot_right[0], top_left[1]:bot_right[1]]

    return cropped_img