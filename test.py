import cv2
import numpy as np
import scipy.misc
import time
import pprint
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("echo", help="provide an image name")
args = parser.parse_args()
num = args.echo
i = 0

def main(image):

    area_lower = 1500
    area_upper = 20000
    ellipse_lower = 1300
    ellipse_upper = 160000

    def find_closest_contour(point, contours):
        x, y = point
        for c in contours:
            cX, cY = compute_centroid(c)
            centerX, centerY = center(c, cX, cY)
            dist = distance((x, y), cX, cY)



    def compute_centroid(c, moments=None):
        if not moments:
            moments = cv2.moments(c)
        if int(moments["m00"]) == 0:
            return
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        return (cX, cY)

    def distance(point, x, y):
        return cv2.pointPolygonTest(point, (x, y), True)

    def center(contour_points, cX, cY):
        return min(contour_points, key=lambda c: abs(distance(c, cX, cY)))

    def endpoint(contour_points, cX, cY):
        return min(contour_points, key=lambda c: distance(c, cX, cY))

    def report(contour, area, cX, cY, closest, ellipse_area, rejected):
        if rejected:
            print('REJECTED')
        print('Contour Detected')
        print('Contour Area:', area)
        print('Centroid', cX, cY)
        print('Closest Point', closest[0], closest[1])
        print('Ellipse Area:', ellipse_area)
        print('---')

    def preprocess(image):
        global i
        image_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_in = np.uint8(cv2.pow(image_in/255.0, 1.4) * 255)
        scipy.misc.imsave("camera_data/uncorrected" + str(i) + ".jpg",  image_in)
        h, s, v = cv2.split(cv2.cvtColor(image_in, cv2.COLOR_RGB2HSV))
        nonSat = s < 180
        disk = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        nonSat = cv2.erode(nonSat.astype(np.uint8), (3,3))
        v2 = v.copy()
        v2[nonSat == 0] = 0
        glare = v2 > 240;
        glare = cv2.dilate(glare.astype(np.uint8), disk);
        corrected = cv2.inpaint(image_in, glare, 5, cv2.INPAINT_NS)
        gray = cv2.cvtColor(image_in, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        scipy.misc.imsave('camera_data/thresh' + str(i) + '.jpg', thresh)
        i += 1
        return thresh

        # Working now
    def process_image(image):

        thresh = preprocess(image)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept = []

        for c in contours:

            M = cv2.moments(c)
            area = cv2.contourArea(c)

            if int(M["m00"]) != 0 and area_lower <= area <= area_upper:

                cX, cY = compute_centroid(c, M)
                closest = np.vstack(center(c, cX, cY)).squeeze()
                Cx, Cy = closest[0], closest[1]
                true_center = (Cx, Cy)

                ellipse = cv2.fitEllipse(c)
                (x,y), (ma,MA), angle = ellipse
                ellipse_aspect = ma/MA
                ellipse_area = (np.pi * ma * MA)/4

                if ellipse_lower < ellipse_area:
                    cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.circle(image, true_center, 10, (0, 0, 0), -1)
                    report(c, area, cX, cY, closest, ellipse_area, False)
                    cv2.ellipse(image, ellipse, (0, 0, 255), 2)
                    cv2.drawContours(image, [c], 0, (0, 255, 255), 2)
                    e = np.vstack(endpoint(c, cX, cY)).squeeze()
                    eX, eY = e[0], e[1]
                    cv2.circle(image, (eX, eY), 10, (0, 170, 0), -1)
                    cv2.line(image, (Cx, Cy), (eX, eY), (255, 0, 0), 10)
                    # rows,cols = image.shape[:2]
                    # line = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01)
                    # [vx,vy,x,y] = line
                    # lefty = int((-x*vy/vx) + y)
                    # righty = int(((cols-x)*vy/vx)+y)
                    # cv2.line(image,(cols-1,righty),(0,lefty),(255, 0, 0),2)
                    kept.append(c)
                else:
                    cv2.drawContours(image,[c],0,(0,139,0),2)
                    cv2.putText(image, "REJECTED", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    report(c, area, cX, cY, closest, ellipse_area, True)
            elif 400 < area < 1200:
                X, Y = compute_centroid(c)
                cv2.circle(image, (X, Y), 10, (233, 0, 0), -1)
                print('area:', area)
                cv2.drawContours(image, [c], 0, (0, 0, 122), 2)


    # def intersection(l1, l2):
    #     [vx, vy, x, y] = [np.asscalar(i) for i in l1]
    #     [vx1, vy1, x1, y1] = [np.asscalar(j) for j in l2]
    #     m1 = vy/vx
    #     m2 = vy1/vx1
    #     a = np.array([[-m1, 1], [-m2, 1]])
    #     b = np.array([[y - (m1)*x], [y1 - (m2)*x1]])
    #     sol = np.linalg.solve(a, b)
    #     return sol
    #
    # def get_intersections(contours):
    #     for c in contours:
    #         l1 = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01)
    #         M = cv2.moments(c)
    #         cX = int(M["m10"] / M["m00"])
    #         cY = int(M["m01"] / M["m00"])
    #         print("Center", (cX, cY))
    #         i = contours.index(c)
    #         others = [c for c in contours if contours.index(c) != i]
    #         sols = []
    #         for o in others:
    #             l2 = cv2.fitLine(o, cv2.DIST_L2,0,0.01,0.01)
    #             x, y = intersection(l1, l2)
    #             cv2.circle(image, (x, y), 10, (255, 255, 255), -1)
    #             sols.append(np.array([np.asscalar(x), np.asscalar(y)]))
    #         print(min(sols, key=lambda s: np.linalg.norm(s - np.array([cX, cY]))))

    process_image(image)

right_image = cv2.imread('images/right' + str(num) + '.jpg')
left_image = cv2.imread('images/left' + str(num) + '.jpg')
main(right_image)
main(left_image)
right = cv2.resize(right_image, (0, 0), fx=0.4, fy=0.4)
left = cv2.resize(left_image, (0, 0), fx=0.4, fy=0.4)
numpy_horizontal = np.hstack((left, right))
cv2.imshow('Numpy Horizontal', numpy_horizontal)
cv2.waitKey(0)
cv2.destroyAllWindows()