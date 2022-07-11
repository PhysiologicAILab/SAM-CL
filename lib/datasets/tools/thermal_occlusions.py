import math
import numpy as np
from PIL import Image, ImageDraw
import random

def generatePolygon(ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts):
    '''
    Start with the centre of the polygon at ctrX, ctrY,
    then creates the polygon by sampling points on a circle around the centre.
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order.
    '''

    irregularity = clip(irregularity, 0, 1) * 2*math.pi / numVerts
    spikeyness = clip(spikeyness, 0, 1) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2*math.pi / numVerts) - irregularity
    upper = (2*math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts):
        tmp = np.random.uniform(lower, upper)
        angleSteps.append(tmp)
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2*math.pi)
    for i in range(numVerts):
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = np.random.uniform(0, 2*math.pi)
    for i in range(numVerts):
        r_i = clip(random.gauss(aveRadius, spikeyness), 0, 2*aveRadius)
        x = ctrX + r_i*math.cos(angle)
        y = ctrY + r_i*math.sin(angle)
        points.append((int(x), int(y)))

        angle = angle + angleSteps[i]

    return points


def clip(x, min, max):
    if(min > max):
        return x
    elif(x < min):
        return min
    elif(x > max):
        return max
    else:
        return x


class ThermOcclusion():
    def __init__(self) -> None:
        self.rad_low = 10
        self.rad_high = 25
        self.irreg_low = 0.0
        self.irreg_high = 1.0
        self.spikeyness_low = 0.0
        self.spikeyness_high = 1.0
        self.verts_low_1 = 2
        self.verts_high_1 = 10
        self.verts_low_2 = 2
        self.verts_high_2 = 100
        self.rand_noise_low = 10    # milli-kelvin
        self.rand_noise_high = 150  # milli-kelvin
        self.population_3 = [1, 2, 3]
        self.weights_3 = [0.34, 0.33, 0.33]

    def gen_occluded_image(self, input_img, low_temp, high_temp):

        img_width, img_height = input_img.shape
        min_dimension = min(img_width, img_height)
        self.ctr_low = 0.1 * min_dimension
        # min is intended for ctr_high - to prevent occlusion getting formed outside the image space
        self.ctr_high = 0.9 * min_dimension
        self.line_width_low = 0.02 * min_dimension
        self.line_width_high = 0.1 * min_dimension

        occ_ch = random.choices(population=self.population_3, weights=self.weights_3)[0]

        black = 0
        white = 1
        crtX = np.random.randint(self.ctr_low, self.ctr_high)
        crtY = np.random.randint(self.ctr_low, self.ctr_high)
        aveRadius = np.random.randint(self.rad_low, self.rad_high)
        irregularity = np.random.uniform(self.irreg_low, self.irreg_high)
        spikeyness = np.random.uniform(self.spikeyness_low, self.spikeyness_high)
        numVerts = np.random.randint(self.verts_low_1, self.verts_high_1)
        rand_noise_magnitude = np.random.randint(self.rand_noise_low, self.rand_noise_high)/1000.0

        verts1 = generatePolygon(
            ctrX=crtX,
            ctrY=crtY,
            aveRadius=aveRadius,
            irregularity=irregularity,
            spikeyness=spikeyness,
            numVerts=numVerts
        )

        occluded_img = Image.new('L', (img_height, img_width), black) #Note that Pillow Image takes height, width dimension, not width, height
        draw = ImageDraw.Draw(occluded_img)  # (rand_noise)  # (im)
        if numVerts > 2:
            draw.polygon(verts1, outline=white, fill=white)
        else:
            line_width = np.random.randint(self.line_width_low, self.line_width_high)
            draw.line(verts1, fill=white, width=line_width)
        occluded_img = np.float64(occluded_img)

        rand_noise1 = (np.random.uniform(low_temp, high_temp)) + (rand_noise_magnitude * (np.random.random(input_img.shape) - 0.5))
        occluded_img = rand_noise1 * occluded_img
        occluded_img[(occluded_img == 0)] = input_img[(occluded_img == 0)]

        if occ_ch == 1:
            pass

        elif occ_ch == 2:
            crtX1 = np.random.randint(self.ctr_low, self.ctr_high)
            crtY1 = np.random.randint(self.ctr_low, self.ctr_high)
            line_width = np.random.randint(self.line_width_low, self.line_width_high)
            # aveRadius = np.random.randint(rad_low, rad_high)
            # irregularity = np.random.uniform(irreg_low, irreg_high)
            # spikeyness = np.random.uniform(spikeyness_low, spikeyness_high)
            # numVerts = np.random.randint(verts_low_2, verts_high_2)

            verts2 = generatePolygon(
                ctrX=crtX1,
                ctrY=crtY1,
                aveRadius=aveRadius,
                irregularity=irregularity,
                spikeyness=spikeyness,
                numVerts=numVerts
            )
            
            im_arr2 = Image.new('L', (img_height, img_width), black)
            draw = ImageDraw.Draw(im_arr2)  # (rand_noise)  # (im)
            draw.polygon(verts2, outline=white, fill=white)

            draw.line([(crtX, crtY), (crtX1, crtY1)], fill=white, width=line_width)
            rand_noise2 = (np.random.uniform(low_temp, high_temp)) + (rand_noise_magnitude * (np.random.random(input_img.shape) - 0.5))

            im_arr2 = np.float64(im_arr2)
            im_arr2 = rand_noise2 * im_arr2
            occluded_img[im_arr2 != 0] = im_arr2[im_arr2 != 0]
            occluded_img[(occluded_img == 0)] = input_img[(occluded_img == 0)]
            # '''

        else:
            crtX = np.random.randint(self.ctr_low, self.ctr_high)
            crtY = np.random.randint(self.ctr_low, self.ctr_high)
            aveRadius = np.random.randint(self.rad_low, self.rad_high)
            irregularity = np.random.uniform(self.irreg_low, self.irreg_high)
            spikeyness = np.random.uniform(self.spikeyness_low, self.spikeyness_high)
            numVerts = np.random.randint(self.verts_low_2, self.verts_high_2)
            rand_noise_magnitude = np.random.randint(self.rand_noise_low, self.rand_noise_high)/1000.0

            verts2 = generatePolygon(
                ctrX=crtX,
                ctrY=crtY,
                aveRadius=aveRadius,
                irregularity=irregularity,
                spikeyness=spikeyness,
                numVerts=numVerts
            )

            im_arr2 = Image.new('L', (img_height, img_width), black)
            draw = ImageDraw.Draw(im_arr2)  # (rand_noise)  # (im)
            if numVerts > 2:
                draw.polygon(verts2, outline=white, fill=white)
            else:
                line_width = np.random.randint(self.line_width_low, self.line_width_high)
                draw.line(verts2, fill=white, width=line_width)
            im_arr2 = np.float64(im_arr2)

            rand_noise2 = (np.random.uniform(low_temp, high_temp)) + (rand_noise_magnitude * (np.random.random(input_img.shape) - 0.5))

            im_arr2 = rand_noise2 * im_arr2
            occluded_img[im_arr2 != 0] = im_arr2[im_arr2 != 0]
            occluded_img[(occluded_img == 0)] = input_img[(occluded_img == 0)]

        return occluded_img