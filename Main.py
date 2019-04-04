import cv2
from skimage.color import rgb2gray
from skimage.data import page
from skimage.filters import threshold_sauvola
import matplotlib.pyplot as plt

from skimage import io
from skimage import img_as_ubyte
from skimage import img_as_float
import time

import numpy as np


class Main:

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = self.read_image()
        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image_threshold = None

    def read_image(self):
        # return cv2.resize(cv2.imread(self.image_path), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        return cv2.imread(self.image_path)

    def sauvola_threshold(self):
        window_size = min(self.image_gray.shape[0], self.image_gray.shape[1]) // 2

        image = img_as_float(self.image_gray)

        thresh_sauvola = threshold_sauvola(image, window_size=15, k=0.34)
        binary_sauvola = image > thresh_sauvola

        self.image_threshold = img_as_ubyte(binary_sauvola)
        cv2.imwrite('image_sauvola.jpg', self.image_threshold)

    def otsu_threshold(self):
        self.image_threshold = cv2.threshold(self.image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        cv2.imwrite('image_otsu.jpg', self.image_threshold)

    # def connected_component(self, nb_components, stats):
    #     def imshow_components(labels):
    #         # Map component labels to hue val
    #         label_hue = np.uint8(179 * labels / np.max(labels))
    #         blank_ch = 255 * np.ones_like(label_hue)
    #         labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    #
    #         # cvt to BGR for display
    #         labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    #
    #         # set bg label to black
    #         labeled_img[label_hue == 0] = 0
    #
    #         cv2.imwrite('labeled.png', labeled_img)
    #
    #     ret, labels = cv2.connectedComponents(self.image_threshold)
    #
    #     imshow_components(labels)
    #
    #     nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(self.image_threshold, connectivity=8)
    #
    #     contours = cv2.findContours(self.image_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    #
    #     print(nb_components)
    #
    #     # sizes = stats[:, -1]
    #     #
    #     # print(output)
    #     # max_label = 2
    #     # max_size = sizes[2]
    #     # for i in range(3, nb_components):
    #     #     if sizes[i] > max_size:
    #     #         max_label = i
    #     #         max_size = sizes[i]
    #     #
    #     # print(max_label)
    #     # img2 = np.zeros(output.shape)
    #     # img2[output == max_label] = 255
    #     # cv2.imwrite("Biggest component.jpg", img2)
    #
    #     def is_text(area, dense, ration, inside):
    #         if area < 6:
    #             return False
    #         if dense < 0.05:
    #             return False
    #         if ration < 0.06:
    #             return False
    #         if len(inside) > 3:
    #             return False
    #
    #         return True
    #
    #     text_cc_features = []
    #     non_text_cc_features = []
    #     for i in range(0, nb_components):
    #         x_min = stats[i, 0]
    #         y_min = stats[i, 1]
    #         x_max = stats[i, 0] + stats[i, 2]
    #         y_max = stats[i, 1] + stats[i, 3]
    #
    #         # the leftmost, the topmost, the rightmost and the lowermost coordinate
    #         cc_bounding_box = [x_min, y_min, x_max, y_max]
    #
    #         cc_area = (self.image_threshold[y_min:y_max, x_min:x_max] == 1).sum()
    #
    #         b_size = stats[i, 2] * stats[i, 3]
    #
    #         cc_dens = cc_area / b_size
    #
    #         cc_ratio = min(stats[i, 2], stats[i, 3]) / max(stats[i, 2], stats[i, 3])
    #
    #         cv2.rectangle(self.image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    #
    #         cc_same_column = []
    #         cc_same_row = []
    #         cc_inside = []
    #         cc_right_neighbors = []
    #         cc_left_neighbors = []
    #         cc_right_nearest_neighbor = 0
    #         cc_left_nearest_neighbor = 0
    #         for j in range(0, nb_components):
    #             if i != j:
    #                 x_min = stats[j, 0]
    #                 y_min = stats[j, 1]
    #                 x_max = stats[j, 0] + stats[j, 2]
    #                 y_max = stats[j, 1] + stats[j, 3]
    #
    #                 if max(cc_bounding_box[0], x_min) - min(cc_bounding_box[2], x_max) < 0:
    #                     cc_same_column.append(j)
    #
    #                 if max(cc_bounding_box[1], y_min) - min(cc_bounding_box[3], y_max) < 0:
    #                     cc_same_row.append(j)
    #
    #                     if x_min - cc_bounding_box[2] > 0:
    #                         if len(cc_right_neighbors) != 0:
    #                             if x_min < stats[cc_right_neighbors[0], 0]:
    #                                 cc_right_neighbors.insert(0, j)
    #                             else:
    #                                 cc_right_neighbors.append(j)
    #                         else:
    #                             cc_right_neighbors.append(j)
    #
    #                     if cc_bounding_box[0] - x_max > 0:
    #                         if len(cc_left_neighbors) != 0:
    #                             if x_max > stats[cc_left_neighbors[0], 0] + stats[cc_left_neighbors[0], 2]:
    #                                 cc_left_neighbors.insert(0, j)
    #                             else:
    #                                 cc_left_neighbors.append(j)
    #
    #                         else:
    #                             cc_left_neighbors.append(j)
    #
    #                 if cc_bounding_box[0] < x_min and cc_bounding_box[1] < y_min \
    #                         and cc_bounding_box[2] > x_max and cc_bounding_box[3] > y_max:
    #                     cc_inside.append(j)
    #
    #                 if len(cc_right_neighbors) != 0:
    #                     cc_right_nearest_neighbor = stats[cc_right_neighbors[0], 0]
    #                     # min_x_min = self.image.shape[1]
    #                     # for component in cc_right_neighbors:
    #                     #     if stats[component, 0] < min_x_min:
    #                     #         min_x_min = stats[component, 0]
    #                     # cc_right_nearest_neighbor = min_x_min
    #                 else:
    #                     cc_right_nearest_neighbor = -1
    #
    #                 if len(cc_left_neighbors) != 0:
    #                     cc_left_nearest_neighbor = stats[cc_left_neighbors[0], 0] + stats[
    #                         cc_left_neighbors[0], 2]
    #                     # max_x_max = 0
    #                     # for component in cc_right_neighbors:
    #                     #     if stats[component, 0] + stats[component, 2] > max_x_max:
    #                     #         max_x_max = stats[component, 0] + stats[component, 2]
    #                     # cc_left_nearest_neighbor = max_x_max
    #                 else:
    #                     cc_left_nearest_neighbor = -1
    #
    #         if is_text(cc_area, cc_dens, cc_ratio, cc_inside):
    #             print("55555555555555")
    #             text_cc_features.append(
    #                 [cc_bounding_box, cc_area, b_size, cc_dens, cc_ratio, cc_same_column, cc_same_row, cc_inside,
    #                  cc_right_neighbors, cc_left_neighbors, cc_right_nearest_neighbor, cc_left_nearest_neighbor])
    #         else:
    #             non_text_cc_features.append(
    #                 [cc_bounding_box, cc_area, b_size, cc_dens, cc_ratio, cc_same_column, cc_same_row, cc_inside,
    #                  cc_right_neighbors, cc_left_neighbors, cc_right_nearest_neighbor, cc_left_nearest_neighbor])
    #
    #     return text_cc_features, non_text_cc_features
    #     # img2 = np.ones(output.shape) * 255
    #     # # img2[output == i] = 0
    #     # cv2.rectangle(img2, (cc_bounding_box[0], cc_bounding_box[1]), (cc_bounding_box[2], cc_bounding_box[3]),
    #     # (0, 255, 0), 3)
    #     # cv2.imwrite("Biggest component_" + str(i) + ".jpg", img2)

    def connected_component2(self, nb_components, stats):

        def is_text(area, dense, ration, inside):
            if area < 6:
                return False
            if dense < 0.05:
                return False
            if ration < 0.06:
                return False
            if len(inside) > 3:
                return False

            return True

        _, contours, _ = cv2.findContours(self.image_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        text_cc_features = []
        non_text_cc_features = []

        for i in range(0, len(contours)):

            x_min, y_min, w0, h0 = cv2.boundingRect(contours[i])
            x_max = x_min + w0
            y_max = y_min + h0

            # the leftmost, the topmost, the rightmost and the lowermost coordinate
            cc_bounding_box = [x_min, y_min, x_max, y_max]

            cc_area = (self.image_threshold[y_min:y_max, x_min:x_max] == 255).sum()

            b_size = w0 * h0

            cc_dens = cc_area / b_size

            cc_ratio = min(w0, h0) / max(w0, h0)

            # cv2.rectangle(self.image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

            cc_same_column = []
            cc_same_row = []
            cc_inside = []
            cc_right_neighbors = []
            cc_left_neighbors = []
            cc_right_nearest_neighbor = 0
            cc_left_nearest_neighbor = 0
            for j in range(0, len(contours)):
                if i != j:
                    x_min, y_min, w0, h0 = cv2.boundingRect(contours[j])
                    x_max = x_min + w0
                    y_max = y_min + h0

                    if max(cc_bounding_box[0], x_min) - min(cc_bounding_box[2], x_max) < 0:
                        cc_same_column.append(j)

                    if max(cc_bounding_box[1], y_min) - min(cc_bounding_box[3], y_max) < 0:
                        cc_same_row.append(j)

                        if x_min - cc_bounding_box[2] > 0:
                            if len(cc_right_neighbors) != 0:
                                if x_min < cv2.boundingRect(contours[cc_right_neighbors[0]])[0]:
                                    cc_right_neighbors.insert(0, j)
                                else:
                                    cc_right_neighbors.append(j)
                            else:
                                cc_right_neighbors.append(j)

                        if cc_bounding_box[0] - x_max > 0:
                            if len(cc_left_neighbors) != 0:
                                if x_max > cv2.boundingRect(contours[cc_left_neighbors[0]])[0] + \
                                        cv2.boundingRect(contours[cc_left_neighbors[0]])[2]:
                                    cc_left_neighbors.insert(0, j)
                                else:
                                    cc_left_neighbors.append(j)

                            else:
                                cc_left_neighbors.append(j)

                    if cc_bounding_box[0] < x_min and cc_bounding_box[1] < y_min \
                            and cc_bounding_box[2] > x_max and cc_bounding_box[3] > y_max:
                        cc_inside.append(j)

                    if len(cc_right_neighbors) != 0:
                        cc_right_nearest_neighbor = cv2.boundingRect(contours[cc_right_neighbors[0]])[0]
                        # min_x_min = self.image.shape[1]
                        # for component in cc_right_neighbors:
                        #     if stats[component, 0] < min_x_min:
                        #         min_x_min = stats[component, 0]
                        # cc_right_nearest_neighbor = min_x_min
                    else:
                        cc_right_nearest_neighbor = -1

                    if len(cc_left_neighbors) != 0:
                        cc_left_nearest_neighbor = cv2.boundingRect(contours[cc_left_neighbors[0]])[0] + \
                                                   cv2.boundingRect(contours[cc_left_neighbors[0]])[2]
                        # max_x_max = 0
                        # for component in cc_right_neighbors:
                        #     if stats[component, 0] + stats[component, 2] > max_x_max:
                        #         max_x_max = stats[component, 0] + stats[component, 2]
                        # cc_left_nearest_neighbor = max_x_max
                    else:
                        cc_left_nearest_neighbor = -1

            if is_text(cc_area, cc_dens, cc_ratio, cc_inside):
                # print("55555555555555")
                text_cc_features.append(
                    [cc_bounding_box, cc_area, b_size, cc_dens, cc_ratio, cc_same_column, cc_same_row, cc_inside,
                     cc_right_neighbors, cc_left_neighbors, cc_right_nearest_neighbor, cc_left_nearest_neighbor])
            else:
                non_text_cc_features.append(
                    [cc_bounding_box, cc_area, b_size, cc_dens, cc_ratio, cc_same_column, cc_same_row, cc_inside,
                     cc_right_neighbors, cc_left_neighbors, cc_right_nearest_neighbor, cc_left_nearest_neighbor])

        return text_cc_features, non_text_cc_features
        # img2 = np.ones(output.shape) * 255
        # # img2[output == i] = 0
        # cv2.rectangle(img2, (cc_bounding_box[0], cc_bounding_box[1]), (cc_bounding_box[2], cc_bounding_box[3]),
        # (0, 255, 0), 3)
        # cv2.imwrite("Biggest component_" + str(i) + ".jpg", img2)

    def print_component(self, text, non_text):
        img2 = np.ones(self.image.shape) * 255
        for c in text:
            cv2.rectangle(self.image, (c[0][0], c[0][1]), (c[0][2], c[0][3]), (0, 0, 255), 3)
        for c in non_text:
            cv2.rectangle(self.image, (c[0][0], c[0][1]), (c[0][2], c[0][3]), (255, 0, 0), 3)

        cv2.imwrite("component.jpg", self.image)

    def run(self):
        self.otsu_threshold()
        text_cc_features, non_text_cc_features = self.connected_component2(None, None)
        self.print_component(text_cc_features, non_text_cc_features)


if __name__ == '__main__':
    IMAGE_PATH = 'images/11.jpg'
    main = Main(IMAGE_PATH)

    start = time.time()
    main.run()
    # main.otsu_threshold()
    # stats = np.array([[1, 1, 4, 4],
    #                   [7, 2, 5, 5],
    #                   [14, 1, 5, 3],
    #                   [14, 6, 5, 3],
    #                   [8, 4, 3, 2]])
    # x = (main.connected_component(5, stats))
    # for xx in x:
    #     print(xx)
    end = time.time()
    print(end - start)
