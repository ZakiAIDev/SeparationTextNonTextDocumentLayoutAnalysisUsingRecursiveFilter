from src.Main import Main
import numpy as np


def test_recursive_segmentation():
    IMAGE_PATH = '../images/3.jpg'
    main = Main(IMAGE_PATH)

    main.sauvola_threshold()

    region = np.array([
        [1, 0, 1, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 0, 0, 0]
    ])

    h_vertical = [2, 3, 2, 0, 0, 4, 2, 0, 0, 0, 4, 2, 3, 1]
    h_vertical_b_level = [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    h_vertical_rle = [1, 3, 0, 2, 1, 2, 0, 3, 1, 4]
    h_vertical_black = [3, 2, 4]
    h_vertical_white = [2, 3]

    h_horizontal = [7, 4, 6, 6]
    h_horizontal_b_level = [1, 1, 1, 1]
    h_horizontal_rle = [1, 4]
    h_horizontal_black = [4]
    h_horizontal_white = []

    features = main.recursive_segmentation(region)

    assert features[0] == h_vertical
    assert features[1] == h_vertical_b_level
    assert features[2] == h_vertical_rle
    assert features[3] == h_vertical_black
    assert features[4] == h_vertical_white

    assert features[5] == h_horizontal
    assert features[6] == h_horizontal_b_level
    assert features[7] == h_horizontal_rle
    assert features[8] == h_horizontal_black
    assert features[9] == h_horizontal_white
# class TestRecursiveSegmentation(object):
#     # make sure to start function name with test
#     # def test_sum(self):
#     #     assert sum(1, 2) == 3
#     #
#     # @pytest.mark.parametrize('num1, num2, expected', [(3, 5, 8), (-2, -2, -4), (-1, 5, 4), (3, -5, -2), (0, 5, 5)])
#     # def test_sum(self, num1, num2, expected):
#     #     assert sum(num1, num2) == expected
#     #
#     @pytest.fixture
#     def get_sum_test_data(self):
#         return [(3, 5, 8), (-2, -2, -4), (-1, 5, 4), (3, -5, -2), (0, 5, 5)]
#
#     def test_sum(self, get_sum_test_data):
#         for data in get_sum_test_data:
#             num1 = data[0]
#             num2 = data[1]
#             expected = data[2]
#             assert sum(num1, num2) == expected
#
#
# def sum(num1, num2):
#     """It returns sum of two numbers"""
#     return num1 + num2
#
