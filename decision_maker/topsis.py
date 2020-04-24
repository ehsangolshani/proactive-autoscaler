from pprint import pprint
from typing import List
from math import sqrt


class TOPSIS:
    def __init__(self, beneficial_column_indicator: List[bool], criteria_weights: list):
        self.criteria_weights = self.normalize_weights(weights=criteria_weights)
        self.beneficial_column_indicator = beneficial_column_indicator

    @staticmethod
    def normalize_weights(weights: list):
        return [weight / sum(weights) for weight in weights]

    # normalize Matrix: TOPSIS algorithm: STEP 1
    @staticmethod
    def calculate_normalized_matrix(matrix: List[list]) -> List[List[float]]:
        # elevation square matrix
        square_matrix = [[y ** 2 for y in row] for row in matrix]

        # get vector of column sum og square matrix
        vector_square_root_sum = [sqrt(sum(x)) for x in zip(*square_matrix)]
        normalized_matrix = [[value / square_root_sum if square_root_sum != 0 else 0.0 for value, square_root_sum in
                              zip(row, vector_square_root_sum)]
                             for row in matrix]
        return normalized_matrix

    # weight normalized matrix: TOPSIS algorithm: STEP 2
    def calculate_weighted_matrix(self, matrix: List[list]):
        weighted_matrix = [[value * weight for value, weight in zip(row, self.criteria_weights)]
                           for row in matrix]
        return weighted_matrix

    def calculate_ideal_points(self, matrix: List[list]):
        ideal_points = [max(x) if self.beneficial_column_indicator[i] is True else min(x)
                        for i, x in enumerate(zip(*matrix))]
        worst_points = [min(x) if self.beneficial_column_indicator[i] is True else max(x)
                        for i, x in enumerate(zip(*matrix))]
        return ideal_points, worst_points

    # distance  from idealpoint: TOPSIS algorithm: STEP 4
    @staticmethod
    def calculate_distances_from_ideal_point(matrix: List[list], ideal_points: list):
        point_distance_from_ideal = [[(value - ideal_point) ** 2 for value, ideal_point in zip(row, ideal_points)]
                                     for row in matrix]
        row_difference_sum = [sum(row) for row in point_distance_from_ideal]
        row_ideal_points_distance = [sqrt(x) for x in row_difference_sum]
        return row_ideal_points_distance

    # distance  from idealpoint: TOPSIS algorithm: STEP 4
    @staticmethod
    def calculate_distances_from_worst_point(matrix: List[list], worst_points: list):
        point_distance_from_worst = [[(value - ideal_point) ** 2 for value, ideal_point in zip(row, worst_points)]
                                     for row in matrix]
        row_difference_sum = [sum(row) for row in point_distance_from_worst]
        row_worst_points_distance = [x ** 0.5 for x in row_difference_sum]
        return row_worst_points_distance

    # relative closeness - TOPSIS algorithm STEP 5
    @staticmethod
    def calculate_performance_scores(distances_from_ideal_point: list, distances_from_worst_point: list):
        performance_scores = [(n / (n + p)) for n, p in zip(distances_from_worst_point, distances_from_ideal_point)]
        return performance_scores

    # returns ranking of decisions in order from best to worst decision
    def rank(self, matrix: List[list]) -> List[int]:
        normalized_matrix = self.calculate_normalized_matrix(matrix=matrix)
        # print('normalized-decision-matrix -->  ')
        # print(normalized_matrix)

        weighted_normalized_matrix = self.calculate_weighted_matrix(matrix=normalized_matrix)
        # print('weighted-normalized-decision-matrix -->  ')
        # print(weighted_normalized_matrix)

        ideal_points, worst_points = self.calculate_ideal_points(matrix=weighted_normalized_matrix)
        distances_from_ideal_point = self.calculate_distances_from_ideal_point(matrix=weighted_normalized_matrix,
                                                                               ideal_points=ideal_points)
        distances_from_worst_point = self.calculate_distances_from_worst_point(matrix=weighted_normalized_matrix,
                                                                               worst_points=worst_points)
        performance_scores = self.calculate_performance_scores(distances_from_ideal_point=distances_from_ideal_point,
                                                               distances_from_worst_point=distances_from_worst_point)
        ranks = sorted(range(len(performance_scores)), key=lambda x: performance_scores[x], reverse=True)
        return ranks
