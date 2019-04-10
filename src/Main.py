import cv2
from skimage.color import rgb2gray
from skimage.data import page
from skimage.filters import threshold_sauvola
import matplotlib.pyplot as plt

from skimage import io
from skimage import img_as_ubyte
from skimage import img_as_float
from itertools import groupby
import time

import numpy as np
import statistics


class Main:

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = self.read_image()
        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image_threshold = None
        self.binary_document = None

    def read_image(self):
        return cv2.resize(cv2.imread(self.image_path), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        # return cv2.imread(self.image_path)

    def sauvola_threshold(self):
        window_size = min(self.image_gray.shape[0], self.image_gray.shape[1]) // 2

        image = img_as_float(self.image_gray)

        thresh_sauvola = threshold_sauvola(image, window_size=15, k=0.34)
        binary_sauvola = image > thresh_sauvola

        self.image_threshold = img_as_ubyte(binary_sauvola)
        self.image_threshold = cv2.bitwise_not(self.image_threshold)

        self.binary_document = self.image_threshold.copy()

        # img = cv2.bitwise_not(self.image_threshold)
        # cv2.imwrite("image_sauvola2.jpg", img)
        cv2.imwrite('image_sauvola.jpg', self.image_threshold)

    def otsu_threshold(self):
        self.image_threshold = cv2.threshold(self.image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        self.image_threshold = cv2.bitwise_not(self.image_threshold)

        self.binary_document = self.image_threshold.copy()

        # cv2.imwrite("binary_document.jpg", self.binary_document)
        cv2.imwrite('image_otsu.jpg', self.image_threshold)

    def connected_component(self, nb_components, stats):
        def imshow_components(labels):
            # Map component labels to hue val
            label_hue = np.uint8(179 * labels / np.max(labels))
            blank_ch = 255 * np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

            # cvt to BGR for display
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

            # set bg label to black
            labeled_img[label_hue == 0] = 0

            cv2.imwrite('labeled.png', labeled_img)

        # ret, labels = cv2.connectedComponents(self.image_threshold)

        # imshow_components(labels)

        # img = cv2.bitwise_not(self.image_threshold)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(self.image_threshold, connectivity=8)

        sorted(stats, key=lambda x: x[1])
        new_stats = [item[1] for item in stats]
        print(nb_components)

        # sizes = stats[:, -1]
        #
        # print(output)
        # max_label = 2
        # max_size = sizes[2]
        # for i in range(3, nb_components):
        #     if sizes[i] > max_size:
        #         max_label = i
        #         max_size = sizes[i]
        #
        # print(max_label)
        # img2 = np.zeros(output.shape)
        # img2[output == max_label] = 255
        # cv2.imwrite("Biggest component.jpg", img2)

        def is_text(area, dense, ration, inside):
            if area < 5:
                return False
            if dense < 0.05:
                return False
            # if ration < 0.06:
            #     return False
            if len(inside) > 3:
                return False

            return True

        text_cc_features = []
        non_text_cc_features = []

        xx = []
        for i in range(2, nb_components):

            cc_features = []
            x_min = stats[i, 0]
            y_min = stats[i, 1]
            x_max = stats[i, 0] + stats[i, 2]
            y_max = stats[i, 1] + stats[i, 3]

            # the leftmost, the topmost, the rightmost and the lowermost coordinate
            cc_bounding_box = [x_min, y_min, x_max, y_max]

            cc_area = (self.image_threshold[y_min:y_max, x_min:x_max] == 255).sum()
            # print(cc_area)
            b_size = stats[i, 2] * stats[i, 3]

            cc_dens = cc_area / b_size

            cc_ratio = min(stats[i, 2], stats[i, 3]) / max(stats[i, 2], stats[i, 3])

            # cv2.rectangle(self.image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

            # cc_same_column = []
            cc_same_row = []
            cc_inside = []
            cc_right_neighbors = []
            cc_left_neighbors = []

            j = new_stats[2:].index(y_min)+2

            while j < nb_components and cc_bounding_box[3] > stats[j][1]:
                if i != j:
                    x_min = stats[j, 0]
                    y_min = stats[j, 1]
                    x_max = stats[j, 0] + stats[j, 2]
                    y_max = stats[j, 1] + stats[j, 3]

                    # if max(cc_bounding_box[0], x_min) - min(cc_bounding_box[2], x_max) < 0:
                    #     cc_same_column.append(j)

                    if max(cc_bounding_box[1], y_min) - min(cc_bounding_box[3], y_max) < 0:
                        cc_same_row.append(j)

                        if x_min - cc_bounding_box[2] > 0:
                            if len(cc_right_neighbors) != 0:
                                if x_min < stats[cc_right_neighbors[0], 0]:
                                    cc_right_neighbors.insert(0, j)
                                else:
                                    cc_right_neighbors.append(j)
                            else:
                                cc_right_neighbors.append(j)

                        if cc_bounding_box[0] - x_max > 0:
                            if len(cc_left_neighbors) != 0:
                                if x_max > stats[cc_left_neighbors[0], 0] + stats[cc_left_neighbors[0], 2]:
                                    cc_left_neighbors.insert(0, j)
                                else:
                                    cc_left_neighbors.append(j)

                            else:
                                cc_left_neighbors.append(j)

                    if cc_bounding_box[0] < x_min and cc_bounding_box[1] < y_min \
                            and cc_bounding_box[2] > x_max and cc_bounding_box[3] > y_max:
                        cc_inside.append(j)
                j += 1
            if len(cc_right_neighbors) != 0:
                cc_right_nearest_neighbor = stats[cc_right_neighbors[0], 0]
                # min_x_min = self.image.shape[1]
                # for component in cc_right_neighbors:
                #     if stats[component, 0] < min_x_min:
                #         min_x_min = stats[component, 0]
                # cc_right_nearest_neighbor = min_x_min
            else:
                cc_right_nearest_neighbor = -1

            if len(cc_left_neighbors) != 0:
                cc_left_nearest_neighbor = stats[cc_left_neighbors[0], 0] + stats[
                    cc_left_neighbors[0], 2]
                # max_x_max = 0
                # for component in cc_right_neighbors:
                #     if stats[component, 0] + stats[component, 2] > max_x_max:
                #         max_x_max = stats[component, 0] + stats[component, 2]
                # cc_left_nearest_neighbor = max_x_max
            else:
                cc_left_nearest_neighbor = -1

            # cc_features.append(cc_same_column)
            # cc_features.append(cc_same_row)
            # cc_features.append(cc_inside)
            # cc_features.append(len(cc_right_neighbors))
            # cc_features.append(len(cc_left_neighbors))
            # cc_features.append(cc_right_nearest_neighbor)
            # cc_features.append(cc_left_nearest_neighbor)
            # xx.append(cc_features)
            if is_text(cc_area, cc_dens, cc_ratio, cc_inside):

                text_cc_features.append(
                    [cc_bounding_box, cc_area, b_size, cc_dens, cc_ratio, cc_same_row, cc_inside,  # cc_same_column
                     cc_right_neighbors, cc_left_neighbors, cc_right_nearest_neighbor, cc_left_nearest_neighbor])
            else:
                self.binary_document[cc_bounding_box[1]:cc_bounding_box[3], cc_bounding_box[0]:cc_bounding_box[2]] = 0
                non_text_cc_features.append(
                    [cc_bounding_box, cc_area, b_size, cc_dens, cc_ratio, cc_same_row, cc_inside,  # cc_same_column
                     cc_right_neighbors, cc_left_neighbors, cc_right_nearest_neighbor, cc_left_nearest_neighbor])

        cv2.imwrite("binary_document.jpg", self.binary_document)

        return text_cc_features, non_text_cc_features
        # img2 = np.ones(output.shape) * 255
        # # img2[output == i] = 0
        # cv2.rectangle(img2, (cc_bounding_box[0], cc_bounding_box[1]), (cc_bounding_box[2], cc_bounding_box[3]),
        # (0, 255, 0), 3)
        # cv2.imwrite("Biggest component_" + str(i) + ".jpg", img2)

    # def connected_component2(self, nb_components, stats):
    #
    #     def is_text(area, dense, ration, inside):
    #         if area < 6:
    #             return False
    #         if dense < 0.15:
    #             return False
    #         if ration < 0.06:
    #             return False
    #         if len(inside) > 4:
    #             return False
    #
    #         return True
    #
    #     _, contours, _ = cv2.findContours(self.image_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     bounding_boxes = [cv2.boundingRect(c) for c in contours]
    #     (contours, boundingBoxes) = zip(*sorted(zip(contours, bounding_boxes),
    #                                             key=lambda b: b[1][1], reverse=False))
    #     print(len(contours))
    #     text_cc_features = []
    #     non_text_cc_features = []
    #
    #     for i in range(1, len(contours)):
    #
    #         x_min, y_min, w0, h0 = cv2.boundingRect(contours[i])
    #         x_max = x_min + w0
    #         y_max = y_min + h0
    #
    #         # the leftmost, the topmost, the rightmost and the lowermost coordinate
    #         cc_bounding_box = [x_min, y_min, x_max, y_max]
    #
    #         cc_area = (self.image_threshold[y_min:y_max, x_min:x_max] == 255).sum()
    #
    #         b_size = w0 * h0
    #
    #         cc_dens = cc_area / b_size
    #
    #         cc_ratio = min(w0, h0) / max(w0, h0)
    #
    #         # cv2.rectangle(self.image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    #
    #         cc_same_column = []
    #         cc_same_row = []
    #         cc_inside = []
    #         cc_right_neighbors = []
    #         cc_left_neighbors = []
    #         cc_right_nearest_neighbor = 0
    #         cc_left_nearest_neighbor = 0
    #         for j in range(1, len(contours)):
    #             if i != j:
    #                 x_min, y_min, w0, h0 = cv2.boundingRect(contours[j])
    #                 x_max = x_min + w0
    #                 y_max = y_min + h0
    #
    #                 if max(cc_bounding_box[0], x_min) - min(cc_bounding_box[2], x_max) < 0:
    #                     cc_same_column.append(j)
    #
    #                 if max(cc_bounding_box[1], y_min) - min(cc_bounding_box[3], y_max) < 0:
    #                     cc_same_row.append(j)
    #
    #                     if x_min - cc_bounding_box[2] > 0:
    #                         if len(cc_right_neighbors) != 0:
    #                             if x_min < cv2.boundingRect(contours[cc_right_neighbors[0]])[0]:
    #                                 cc_right_neighbors.insert(0, j)
    #                             else:
    #                                 cc_right_neighbors.append(j)
    #                         else:
    #                             cc_right_neighbors.append(j)
    #
    #                     if cc_bounding_box[0] - x_max > 0:
    #                         if len(cc_left_neighbors) != 0:
    #                             if x_max > cv2.boundingRect(contours[cc_left_neighbors[0]])[0] + \
    #                                     cv2.boundingRect(contours[cc_left_neighbors[0]])[2]:
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
    #                     cc_right_nearest_neighbor = cv2.boundingRect(contours[cc_right_neighbors[0]])[0]
    #                     # min_x_min = self.image.shape[1]
    #                     # for component in cc_right_neighbors:
    #                     #     if stats[component, 0] < min_x_min:
    #                     #         min_x_min = stats[component, 0]
    #                     # cc_right_nearest_neighbor = min_x_min
    #                 else:
    #                     cc_right_nearest_neighbor = -1
    #
    #                 if len(cc_left_neighbors) != 0:
    #                     cc_left_nearest_neighbor = cv2.boundingRect(contours[cc_left_neighbors[0]])[0] + \
    #                                                cv2.boundingRect(contours[cc_left_neighbors[0]])[2]
    #                     # max_x_max = 0
    #                     # for component in cc_right_neighbors:
    #                     #     if stats[component, 0] + stats[component, 2] > max_x_max:
    #                     #         max_x_max = stats[component, 0] + stats[component, 2]
    #                     # cc_left_nearest_neighbor = max_x_max
    #                 else:
    #                     cc_left_nearest_neighbor = -1
    #
    #         if is_text(cc_area, cc_dens, cc_ratio, cc_inside):
    #             # print("55555555555555")
    #             text_cc_features.append(
    #                 [cc_bounding_box, cc_area, b_size, cc_dens, cc_ratio, cc_same_column, cc_same_row, cc_inside,
    #                  cc_right_neighbors, cc_left_neighbors, cc_right_nearest_neighbor, cc_left_nearest_neighbor])
    #         else:
    #             self.binary_document[cc_bounding_box[1]:cc_bounding_box[3], cc_bounding_box[0]:cc_bounding_box[2]] = 0
    #             non_text_cc_features.append(
    #                 [cc_bounding_box, cc_area, b_size, cc_dens, cc_ratio, cc_same_column, cc_same_row, cc_inside,
    #                  cc_right_neighbors, cc_left_neighbors, cc_right_nearest_neighbor, cc_left_nearest_neighbor])
    #
    #     cv2.imwrite("binary_document.jpg", self.binary_document)
    #     return text_cc_features, non_text_cc_features
    #     # img2 = np.ones(output.shape) * 255
    #     # # img2[output == i] = 0
    #     # cv2.rectangle(img2, (cc_bounding_box[0], cc_bounding_box[1]), (cc_bounding_box[2], cc_bounding_box[3]),
    #     # (0, 255, 0), 3)
    #     # cv2.imwrite("Biggest component_" + str(i) + ".jpg", img2)

    def recursive_segmentation(self, region):
        def rle(histogram):
            rle_list = []
            length_zero = 0
            length_one = 0

            for x in histogram:
                if x == 0:
                    length_zero += 1
                    if length_one != 0:
                        rle_list.append(1)
                        rle_list.append(length_one)
                        length_one = 0
                else:
                    length_one += 1
                    if length_zero != 0:
                        rle_list.append(0)
                        rle_list.append(length_zero)
                        length_zero = 0

            if length_one != 0:
                rle_list.append(1)
                rle_list.append(length_one)
            else:
                rle_list.append(0)
                rle_list.append(length_zero)

            return rle_list

        # r_bounding_box = [0, 0, 0, 0]
        #
        # r_binary_portion_image = self.binary_document[r_bounding_box[1]: r_bounding_box[3],
        #                          r_bounding_box[0]: r_bounding_box[2]]
        #
        # img = cv2.bitwise_not(r_binary_portion_image)
        nb_components, output, r_cc_list, centroids = cv2.connectedComponentsWithStats(region, connectivity=8)

        # r_vertical_histogram = r_binary_portion_image.sum(axis=1).flatten().tolist()[0]
        # r_vertical_histogram_b_level = [1 if x > 0 else 0 for x in r_vertical_histogram]
        #
        # r_horizontal_histogram = r_binary_portion_image.sum(axis=0).flatten().tolist()[0]
        # r_horizontal_histogram_b_level = [1 if x > 0 else 0 for x in r_horizontal_histogram]

        r_vertical_histogram = region.sum(axis=1).flatten().tolist()
        r_vertical_histogram_b_level = [1 if x > 0 else 0 for x in r_vertical_histogram]

        r_horizontal_histogram = region.sum(axis=0).flatten().tolist()
        r_horizontal_histogram_b_level = [1 if x > 0 else 0 for x in r_horizontal_histogram]

        rle_vertical = rle(r_vertical_histogram_b_level)
        rle_horizontal = rle(r_horizontal_histogram_b_level)

        r_vertical_black = []
        r_vertical_white = []
        for i in range(0, len(rle_vertical) - 1, 2):
            if rle_vertical[i] == 0:
                r_vertical_white.append(rle_vertical[i + 1])
            else:
                r_vertical_black.append(rle_vertical[i + 1])

        r_horizontal_black = []
        r_horizontal_white = []
        for i in range(0, len(rle_horizontal) - 1, 2):
            if rle_horizontal[i] == 0:
                r_horizontal_white.append(rle_horizontal[i + 1])
            else:
                r_horizontal_black.append(rle_horizontal[i + 1])

        r_vertical_black_median = statistics.median(r_vertical_black) if r_vertical_black else 0
        r_vertical_white_median = statistics.median(r_vertical_white) if r_vertical_white else 0
        r_horizontal_black_median = statistics.median(r_horizontal_black) if r_horizontal_black else 0
        r_horizontal_white_median = statistics.median(r_horizontal_white) if r_horizontal_white else 0

        r_vertical_black_variance = statistics.variance(r_vertical_black) if len(r_vertical_black) > 1 else 0
        r_vertical_white_variance = statistics.variance(r_vertical_white) if len(r_vertical_white) > 1 else 0
        r_horizontal_black_variance = statistics.variance(r_horizontal_black) if len(r_horizontal_black) > 1 else 0
        r_horizontal_white_variance = statistics.variance(r_horizontal_white) if len(r_horizontal_white) > 1 else 0

        rs_features = {'HVertical': r_vertical_histogram,
                       'LVerticalBlack': r_vertical_black,
                       'LVerticalWhite': r_vertical_white,
                       'HHorizontal': r_horizontal_histogram,
                       'LHorizontalBlack': r_horizontal_black,
                       'LHorizontalWhite': r_horizontal_white,
                       'blevel1': r_vertical_histogram_b_level,
                       'rle_vertical': rle_vertical,
                       'blevel2': r_horizontal_histogram_b_level,
                       'rle_horizontal': rle_horizontal,
                       'M_LVB': r_vertical_black_median,
                       'M_LVW': r_vertical_white_median,
                       'M_LHB': r_horizontal_black_median,
                       'M_LHW': r_horizontal_white_median,
                       'V_LVB': r_vertical_black_variance,
                       'V_LVW': r_vertical_white_variance,
                       'V_LHB': r_horizontal_black_variance,
                       'V_LHW': r_horizontal_white_variance,
                       'nb_components': nb_components}

        # rs_features = [r_vertical_histogram, r_vertical_histogram_b_level, rle_vertical, r_vertical_black,
        #                r_vertical_white, r_horizontal_histogram, r_horizontal_histogram_b_level, rle_horizontal,
        #                r_horizontal_black, r_horizontal_white, r_vertical_black_median, r_vertical_white_median,
        #                r_horizontal_black_median, r_horizontal_white_median, r_vertical_black_variance,
        #                r_vertical_white_variance, r_horizontal_black_variance, r_horizontal_white_variance]

        return rs_features

    def algo(self):
        t_variance = 1.3
        i = 0
        open_list = [self.binary_document]
        vertical_homogeonus_ = []
        horizontal_homogeonus = []
        while open_list:
            region = open_list.pop(0)
            rs_features = self.recursive_segmentation(region)
            # for key, val in rs_features.items():
            #     print(key)
            #     print(val)
            if rs_features['V_LVB'] > t_variance or rs_features['V_LVW'] > t_variance:
                # print(rs_features['blevel1'])
                # step 2
                if rs_features['V_LVB'] > rs_features['V_LVW']:
                    max_Line = max(rs_features['LVerticalBlack'])
                    if max_Line > rs_features['M_LVB']:
                        rle_vertical = rs_features['rle_vertical']
                        count = 0
                        for i in range(0, len(rle_vertical) - 1, 2):
                            if rle_vertical[i] != 1:
                                count += rle_vertical[i + 1]
                            elif rle_vertical[i + 1] != max_Line:
                                count += rle_vertical[i + 1]
                            else:
                                break

                        new_region1 = region[:count, :]
                        new_region2 = region[count: count + max_Line, :]
                        new_region3 = region[count + max_Line:, :]
                        if cv2.connectedComponentsWithStats(new_region1, connectivity=8)[0] > 2:
                            # print(cv2.connectedComponentsWithStats(new_region1, connectivity=8)[0])
                            open_list.append(new_region1)
                        if cv2.connectedComponentsWithStats(new_region2, connectivity=8)[0] > 2:
                            # print(cv2.connectedComponentsWithStats(new_region2, connectivity=8)[0])
                            open_list.append(new_region2)
                        if cv2.connectedComponentsWithStats(new_region3, connectivity=8)[0] > 2:
                            # print(cv2.connectedComponentsWithStats(new_region3, connectivity=8)[0])
                            open_list.append(new_region3)
                    else:
                        # step 1
                        vertical_homogeonus_.append(region)
                        # cv2.imwrite('homogeneous_' + str(i) + ".jpg", region)
                        i += 1
                else:
                    max_Line = max(rs_features['LVerticalWhite'])
                    if max_Line > rs_features['M_LVW']:
                        rle_vertical = rs_features['rle_vertical']
                        count = 0
                        for i in range(0, len(rle_vertical) - 1, 2):
                            if rle_vertical[i] != 0:
                                count += rle_vertical[i + 1]
                            elif rle_vertical[i + 1] != max_Line:
                                count += rle_vertical[i + 1]
                            else:
                                break

                        new_region1 = region[:count, :]
                        new_region2 = region[count + max_Line:, :]
                        if cv2.connectedComponentsWithStats(new_region1, connectivity=8)[0] > 2:
                            # print(cv2.connectedComponentsWithStats(new_region1, connectivity=8)[0])
                            open_list.append(new_region1)
                        if cv2.connectedComponentsWithStats(new_region2, connectivity=8)[0] > 2:
                            # print(cv2.connectedComponentsWithStats(new_region2, connectivity=8)[0])
                            open_list.append(new_region2)

                    else:
                        # step 1
                        if region.shape[0] != 0:
                            vertical_homogeonus_.append(region)
                        # cv2.imwrite('homogeneous_' + str(i) + ".jpg", region)
                        i += 1
            else:
                # step 4
                # cv2.imwrite('homogeneous_' + str(i) + ".jpg", region)
                if region.shape[0] != 0:
                    vertical_homogeonus_.append(region)
                i += 1

        horizontal_homogeonus.extend(vertical_homogeonus_)
        while vertical_homogeonus_:
            region = vertical_homogeonus_.pop(0)
            rs_features = self.recursive_segmentation(region)
            # for key, val in rs_features.items():
            #     print(key)
            #     print(val)
            if rs_features['V_LHB'] > t_variance or rs_features['V_LHW'] > t_variance:
                # print(rs_features['blevel1'])
                # step 2
                if rs_features['V_LHB'] > rs_features['V_LHW']:
                    max_Line = max(rs_features['LHorizontalBlack'])
                    if max_Line > rs_features['M_LHB']:
                        rle_horizontal = rs_features['rle_horizontal']
                        count = 0
                        for i in range(0, len(rle_horizontal) - 1, 2):
                            if rle_horizontal[i] != 1:
                                count += rle_horizontal[i + 1]
                            elif rle_horizontal[i + 1] != max_Line:
                                count += rle_horizontal[i + 1]
                            else:
                                break

                        new_region1 = region[:, :count]
                        new_region2 = region[:, count: count + max_Line]
                        new_region3 = region[:, count + max_Line:]

                        if cv2.connectedComponentsWithStats(new_region1, connectivity=8)[0] > 2:
                            # print(cv2.connectedComponentsWithStats(new_region1, connectivity=8)[0])
                            vertical_homogeonus_.append(new_region1)
                        if cv2.connectedComponentsWithStats(new_region2, connectivity=8)[0] > 2:
                            # print(cv2.connectedComponentsWithStats(new_region2, connectivity=8)[0])
                            vertical_homogeonus_.append(new_region2)
                        if cv2.connectedComponentsWithStats(new_region3, connectivity=8)[0] > 2:
                            # print(cv2.connectedComponentsWithStats(new_region3, connectivity=8)[0])
                            vertical_homogeonus_.append(new_region3)
                    else:
                        # step 1
                        horizontal_homogeonus.append(region)
                        # cv2.imwrite('homogeneous_' + str(i) + ".jpg", region)
                        i += 1
                else:
                    max_Line = max(rs_features['LHorizontalWhite'])
                    if max_Line > rs_features['M_LHW']:
                        rle_horizontal = rs_features['rle_horizontal']
                        count = 0
                        for i in range(0, len(rle_horizontal) - 1, 2):
                            if rle_horizontal[i] != 0:
                                count += rle_horizontal[i + 1]
                            elif rle_horizontal[i + 1] != max_Line:
                                count += rle_horizontal[i + 1]
                            else:
                                break

                        new_region1 = region[:, :count]
                        new_region2 = region[:, count + max_Line:]
                        if cv2.connectedComponentsWithStats(new_region1, connectivity=8)[0] > 2:
                            # print(cv2.connectedComponentsWithStats(new_region1, connectivity=8)[0])
                            vertical_homogeonus_.append(new_region1)
                        if cv2.connectedComponentsWithStats(new_region2, connectivity=8)[0] > 2:
                            # print(cv2.connectedComponentsWithStats(new_region2, connectivity=8)[0])
                            vertical_homogeonus_.append(new_region2)

                    else:
                        # step 1
                        if region.shape[0] != 0:
                            horizontal_homogeonus.append(region)
                        # cv2.imwrite('homogeneous_' + str(i) + ".jpg", region)
                        i += 1
            else:
                # step 4
                # cv2.imwrite('homogeneous_' + str(i) + ".jpg", region)
                if region.shape[0] != 0:
                    horizontal_homogeonus.append(region)
                i += 1

        return horizontal_homogeonus

    def connected_component2(self, region):

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(region, connectivity=8)

        w_s = []
        omega_h = []
        omega_area = []
        omega_w = []
        cc_features = []
        for i in range(2, nb_components):

            x_min = stats[i, 0]
            y_min = stats[i, 1]
            x_max = stats[i, 0] + stats[i, 2]
            y_max = stats[i, 1] + stats[i, 3]

            # the leftmost, the topmost, the rightmost and the lowermost coordinate
            cc_bounding_box = [x_min, y_min, x_max, y_max]

            cc_area = (self.image_threshold[y_min:y_max, x_min:x_max] == 1).sum()

            omega_area.append(cc_area)
            omega_h.append(y_max - y_min)
            omega_w.append(x_max - x_min)

            cc_features.append([cc_area, y_max - y_min, x_max - x_min])
            cc_same_column = []
            cc_same_row = []
            cc_inside = []
            cc_right_neighbors = []
            cc_left_neighbors = []

            for j in range(2, nb_components):
                if i != j:
                    x_min = stats[j, 0]
                    y_min = stats[j, 1]
                    x_max = stats[j, 0] + stats[j, 2]
                    y_max = stats[j, 1] + stats[j, 3]

                    if max(cc_bounding_box[0], x_min) - min(cc_bounding_box[2], x_max) < 0:
                        cc_same_column.append(j)

                    if max(cc_bounding_box[1], y_min) - min(cc_bounding_box[3], y_max) < 0:
                        cc_same_row.append(j)

                        if x_min - cc_bounding_box[2] > 0:
                            if len(cc_right_neighbors) != 0:
                                if x_min < stats[cc_right_neighbors[0], 0]:
                                    cc_right_neighbors.insert(0, j)
                                else:
                                    cc_right_neighbors.append(j)
                            else:
                                cc_right_neighbors.append(j)

                        if cc_bounding_box[0] - x_max > 0:
                            if len(cc_left_neighbors) != 0:
                                if x_max > stats[cc_left_neighbors[0], 0] + stats[cc_left_neighbors[0], 2]:
                                    cc_left_neighbors.insert(0, j)
                                else:
                                    cc_left_neighbors.append(j)
                            else:
                                cc_left_neighbors.append(j)

                    if cc_bounding_box[0] < x_min and cc_bounding_box[1] < y_min \
                            and cc_bounding_box[2] > x_max and cc_bounding_box[3] > y_max:
                        cc_inside.append(j)

            if len(cc_right_neighbors) != 0:
                cc_right_nearest_neighbor = stats[cc_right_neighbors[0], 0]
            else:
                cc_right_nearest_neighbor = -1

            if len(cc_left_neighbors) != 0:
                cc_left_nearest_neighbor = stats[cc_left_neighbors[0], 0] + stats[
                    cc_left_neighbors[0], 2]
            else:
                cc_left_nearest_neighbor = -1

            w_s_left = x_min - cc_left_nearest_neighbor
            w_s_right = cc_right_nearest_neighbor - x_max
            if w_s_right > 0:
                w_s.append(w_s_right)
        features = {'omega_area': omega_area, 'omega_h': omega_h, 'omega_w': omega_w,
                    'w_s_left': w_s_left,
                    'w_s_right': w_s_right,
                    'w_s': w_s}

        return cc_features, features

    def recursive_filter(self, homogeneous_region):

        def max_median_filter(omega_area, omega_h, omega_w, area, h, w):
            media_area = statistics.median(omega_area) if omega_area else 0
            media_h = statistics.median(omega_h) if omega_h else 0
            media_w = statistics.median(omega_w) if omega_w else 0

            mean_area = statistics.mean(omega_area) if omega_area else 0
            mean_h = statistics.mean(omega_h) if omega_h else 0
            mean_w = statistics.mean(omega_w) if omega_w else 0

            k1 = max(mean_area / media_area, media_area / mean_area)
            k2 = max(mean_h / media_h, media_h / mean_h)
            k3 = max(mean_w / media_w, media_w / mean_w)

            if area == max(omega_area) and area > k1 * media_area:
                return False

            if h == max(omega_h) and h > k2 * media_h or w == max(omega_w) and h > k3 * media_w:
                return False

        for region in homogeneous_region:
            cc_features, features = self.connected_component2(region)
            for cc_feature in cc_features:
                max_median_filter(features['omega_area'], features['omega_h'], features['omega_w'],
                                  cc_feature[0], cc_feature[1], cc_feature[2])

    def print_component(self, text, non_text):
        img2 = np.ones(self.image.shape) * 255
        for c in text:
            cv2.rectangle(self.image, (c[0][0], c[0][1]), (c[0][2], c[0][3]), (0, 0, 255), 3)
        for c in non_text:
            cv2.rectangle(self.image, (c[0][0], c[0][1]), (c[0][2], c[0][3]), (255, 0, 0), 3)

        cv2.imwrite("component.jpg", self.image)

    def print_homogeneous(self, homogeneous_list):

        for i, homogeneous in enumerate(homogeneous_list):
            cv2.imwrite('homogeneous_' + str(i) + ".jpg", homogeneous)

    def run(self):
        self.sauvola_threshold()
        text_cc_features, non_text_cc_features = self.connected_component(None, None)
        self.print_component(text_cc_features, non_text_cc_features)
        homogeneous_list = self.algo()
        self.print_homogeneous(homogeneous_list)
        # self.recursive_filter(homogeneous_list)


if __name__ == '__main__':
    IMAGE_PATH = './images/12.jpg'
    main = Main(IMAGE_PATH)

    start = time.time()
    main.run()
    # main.otsu_threshold()
    # stats = np.array([[1, 1, 4, 4],
    #                   [7, 2, 5, 5],
    #                   [14, 1, 5, 3],
    #                   [14, 6, 5, 3],
    #                   [8, 4, 3, 2]])
    # x = (main.connected_component(5, stats))[0]
    # for xx in x:
    #     print(xx)
    end = time.time()
    print(end - start)
