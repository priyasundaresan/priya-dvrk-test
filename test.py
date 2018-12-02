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

    area_lower = 3000
    area_upper = 20000
    ellipse_lower = 1300
    ellipse_upper = 180000

    def find_closest_contour(point, contours):
        x, y = tuple(point)
        point = point.reshape(1, 2)
        return min(contours, key=lambda c: distance_pt_to_contour(c, x, y))
        # for c in contours:
        #     cX, cY = compute_centroid(c)
        #     CX, CY = list(center(c, cX, cY).squeeze())
        #     dist = distance(point, CX, CY)


    def compute_centroid(c, moments=None):
        if not moments:
            moments = cv2.moments(c)
        if int(moments["m00"]) == 0:
            return
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        return (cX, cY)

    def distance(point, x, y):
        # print('computing distance', point, x, y)
        return cv2.pointPolygonTest(point, (x, y), True)

    def distance_pt_to_contour(contour, x, y):
        """Computes the distance from a point to the center (not centroid) of contour"""
        cX, cY = compute_centroid(contour)
        true_center = center(contour, cX, cY)
        return abs(distance(true_center, x, y))

    def find_residual(contours, Cx, Cy):
        return min(contours, key=lambda c: distance_pt_to_contour(c, Cx, Cy))

    def center(contour_points, cX, cY):
        return min(contour_points, key=lambda c: abs(distance(c, cX, cY)))

    def endpoints(contour_points, cX, cY):
        sorted_points = sorted([list(i.squeeze()) for i in contour_points])
        e1 = tuple(sorted_points[0])
        e2 = tuple(sorted_points[-1])
        return (e1, e2)

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
        image_in = cv2.blur(image_in, (3, 3))
        corrected = np.uint8(cv2.pow(image_in/255.0, 1.4) * 255)
        gray = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        i += 1
        return thresh

        # Working now
    def process_image(image):

        thresh = preprocess(image)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept = []

        residuals = [c for c in contours if 600 < cv2.contourArea(c) < 2000]

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

                if ellipse_lower < ellipse_area < ellipse_upper:
                    cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.circle(image, (cX, cY), 10, (255, 0, 0), -1)
                    cv2.circle(image, true_center, 10, (0, 0, 0), -1)
                    report(c, area, cX, cY, closest, ellipse_area, False)
                    cv2.ellipse(image, ellipse, (0, 0, 255), 2)
                    cv2.drawContours(image, [c], 0, (0, 255, 255), 2)

                    residual = find_closest_contour(closest, residuals)
                    r_cX, r_cY = compute_centroid(residual)
                    cv2.drawContours(image, [residual], 0, (255, 255, 255), 2)
                    cv2.circle(image, (r_cX, r_cY), 10, (255, 0, 0), -1)
                    rows,cols = image.shape[:2]
                    [vx,vy,x,y] = cv2.fitLine(residual, cv2.DIST_L2,0,0.01,0.01)
                    vx, vy = np.asscalar(vx), np.asscalar(vy)
                    print('Residual:', r_cX, r_cY)
                    print('Larger:', true_center)
                    print('Vx:', vx, 'Vy:', vy)
                    lefty = int((-x*vy/vx) + y)
                    righty = int(((cols-x)*vy/vx)+y)
                    cv2.line(image,(cols-1,righty),(0,lefty),(0,255,0),2)
                    dx, dy = vx, vy
                    if abs(r_cX - (Cx + dx)) < abs(r_cX - Cx) and abs(r_cY - (Cy + dy)) < abs(r_cY - Cy):
                        dx, dy = -dx, -dy
                    pull_x = int(Cx + 100*dx)
                    pull_y = int(Cy + 100*dy)
                    cv2.circle(image, (pull_x, pull_y), 10, (255, 255, 100), -1)
                    cv2.line(image, (true_center), (pull_x, pull_y), (0, 255, 0), 2)
                    # cv2.line(image, residual_e1, residual_e2, (0,255,0), 2)
                    # cv2.circle(image, pull_point, 10, (255, 170, 0), -1)


                else:
                    cv2.drawContours(image,[c],0,(0,139,0),2)
                    cv2.putText(image, "REJECTED", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    report(c, area, cX, cY, closest, ellipse_area, True)

    process_image(image)

left_image = cv2.imread('images/left' + str(num) + '.jpg')
main(left_image)
left = cv2.resize(left_image, (0, 0), fx=0.4, fy=0.4)
cv2.imshow('Left', left)
cv2.waitKey(0)
cv2.destroyAllWindows()