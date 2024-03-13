"""
Утилиты для решения проблемы упаковки
"""
import random as rd
from copy import deepcopy
from typing import List

import numpy as np
from nptyping import NDArray, Int, Shape


def boxes_generator(
    bin_size: List[int], num_items: int = 64, seed: int = 42
) -> List[List[int]]:
    """Генерирует экземпляры с упаковкой 2D и 3D контейнеров

    Параметры
    ----------
    num_items: int, необязательно
        Количество генерируемых коробок (по умолчанию = 64)
    bin_size: List[int], необязательно (по умолчанию = [10,10,10])
        Список длиной 2 или 3 с размерами контейнера (по умолчанию = (10,10,10))
    seed: int, необязательно
        seed для генератора случайных чисел (по умолчанию = 42)

    Возвращается
    -------
    List[List[int]]
    Список элементов длины num_items с размерами случайно сгенерированных коробок.
    """
    rd.seed(seed)

    dim = len(bin_size)
    # инициализация списка элементов
    item_sizes = [bin_size]

    while len(item_sizes) < num_items:
        # выбераем элемент случайным образом по его объему
        box_vols = [np.prod(np.array(box_size)) for box_size in item_sizes]
        index = rd.choices(list(range(len(item_sizes))), weights=box_vols, k=1)[0]
        box0_size = item_sizes.pop(index)

        # выбор оси (x или y для 2D или x, y, z для 3D) случайным образом по длине ребра элемента
        axis = rd.choices(list(range(dim)), weights=box0_size, k=1)[0]
        len_edge = box0_size[axis]
        while len_edge == 1:
            axis = rd.choices(list(range(dim)), weights=box0_size, k=1)[0]
            len_edge = box0_size[axis]

        # выбор точки разделения вдоль этой оси
        if len_edge == 2:
            split_point = 1
        else:
            dist_edge_center = [abs(x - len_edge / 2) for x in range(1, len_edge)]
            weights = np.reciprocal(np.asarray(dist_edge_center) + 1)
            split_point = rd.choices(list(range(1, len_edge)), weights=weights, k=1)[0]

        # разделение box0 на box1 и box2 в точке разделения на выбранной оси
        box1 = deepcopy(box0_size)
        box2 = deepcopy(box0_size)
        box1[axis] = split_point
        box2[axis] = len_edge - split_point
        assert (np.prod(box1) + np.prod(box2)) == np.prod(box0_size)

        # поворот коробки по самой длинной стороне
        # добавление коробки в список элементов
        # box1.sort(reverse=True)
        # box2.sort(reverse=True)
        item_sizes.extend([box1, box2])

    return item_sizes


def generate_vertices(
    cuboid_len_edges: NDArray, cuboid_position: NDArray
) -> NDArray[Shape["3, 8"], Int]:
    """Генерирует вершины коробки или контейнера в правильном формате для построения графика

    Параметры
    ----------
    cuboid_position: list[int]
          Список с координатами задней нижней левой вершины коробки или контейнера
    cuboid_len_edges: list[int]
        Список с размерами коробки или контейнера

    Возвращается
    -------
    np.nd.array(np.int32)
    Массив формы (3,8) с координатами вершин прямоугольника или контейнера
    """
    # Генерация списка вершин, добавляем длины ребер к координатам
    v0 = cuboid_position
    v0 = np.asarray(v0, dtype=np.int32)
    v1 = v0 + np.asarray([cuboid_len_edges[0], 0, 0], dtype=np.int32)
    v2 = v0 + np.asarray([0, cuboid_len_edges[1], 0], dtype=np.int32)
    v3 = v0 + np.asarray([cuboid_len_edges[0], cuboid_len_edges[1], 0], dtype=np.int32)
    v4 = v0 + np.asarray([0, 0, cuboid_len_edges[2]], dtype=np.int32)
    v5 = v1 + np.asarray([0, 0, cuboid_len_edges[2]], dtype=np.int32)
    v6 = v2 + np.asarray([0, 0, cuboid_len_edges[2]], dtype=np.int32)
    v7 = v3 + np.asarray([0, 0, cuboid_len_edges[2]], dtype=np.int32)
    vertices = np.vstack((v0, v1, v2, v3, v4, v5, v6, v7))
    return vertices


def interval_intersection(a: List[int], b: List[int]) -> bool:
    """Проверяет, имеют ли два открытых интервала с целочисленными конечными точками непустое пересечение.

    Параметры
    ----------
    a: List[int]
        Список с началом и концом первого интервала
    b: List[int]
        Список с началом и концом второго интервала

    Возвращается
    -------
    тип bool
    True, если интервалы пересекаются, False в противном случае
    """
    assert a[1] > a[0], "a[1] must be greater than a[0]"
    assert b[1] > b[0], "b[1] must be greater than b[0]"
    return min(a[1], b[1]) - max(a[0], b[0]) > 0


def cuboids_intersection(cuboid_a: List[int], cuboid_b: List[int]) -> bool:
    """Проверяет, пересекаются ли два куба.

    Параметры
    ----------
    cuboid_a: List[int]
        Список [x_min_a, y_min, z_min_a, x_max_a, y_max_a, z_max_a]
        с координатами начала и конца первого куба на каждой оси

    cuboid_a: List[int]
        Список [x_min_b, y_min_b, z_min_b, x_max_b, y_max_b, z_max_b]
        с начальными и конечными координатами второго куба на каждой оси

    Возвращается
    -------
    тип bool
    True, если кубоиды пересекаются, False в противном случае
    """
    assert len(cuboid_a) == 6, "cuboid_a must be a list of length 6"
    assert len(cuboid_b) == 6, "cuboid_b must be a list of length 6"

    # Проверка координаты задней нижней левой вершины первого куба
    assert np.all(
        np.less_equal([0, 0, 0], cuboid_a[:3])
    ), "cuboid_a must have nonnegative coordinates"
    assert np.all(
        np.less_equal([0, 0, 0], cuboid_b[:3])
    ), "cuboid_b must have non-negative coordinates"

    assert np.all(
        np.less(cuboid_a[:3], cuboid_a[3:])
    ), "cuboid_a must have non-zero volume"

    assert np.all(
        np.less(cuboid_b[:3], cuboid_b[3:])
    ), "cuboid_b must have non-zero volume"

    inter = [
        interval_intersection([cuboid_a[0], cuboid_a[3]], [cuboid_b[0], cuboid_b[3]]),
        interval_intersection([cuboid_a[1], cuboid_a[4]], [cuboid_b[1], cuboid_b[4]]),
        interval_intersection([cuboid_a[2], cuboid_a[5]], [cuboid_b[2], cuboid_b[5]]),
    ]

    return np.all(inter)


def cuboid_fits(cuboid_a: List[int], cuboid_b: List[int]) -> bool:
    """Проверяет, вписывается ли cuboid_b в cuboid_a.
    Параметры
    ----------
    cuboid_a: List[int]
        Список [x_min_a, y_min, z_min_a, x_max_a, y_max_a, z_max_a]
        с координатами начала и конца первого куба на каждой оси
    cuboid_b: List[int]
        Список [x_min_b, y_min_b, z_min_b, x_max_b, y_max_b, z_max_b]
        с координатами начала и конца второго куба на каждой оси
    Возвращается
    -------
    тип bool
    True, если cuboid_b вписывается в cuboid_a, False в противном случае
    """
    assert len(cuboid_a) == 6, "cuboid_a must be a list of length 3"
    assert len(cuboid_b) == 6, "cuboid_b must be a list of length 3"

    assert len(cuboid_a) == 6, "cuboid_a must be a list of length 6"
    assert len(cuboid_b) == 6, "cuboid_b must be a list of length 6"


    assert np.all(
        np.less_equal([0, 0, 0], cuboid_a[:3])
    ), "cuboid_a must have non-negative coordinates"
    assert np.all(
        np.less_equal([0, 0, 0], cuboid_b[:3])
    ), "cuboid_b must have non-negative coordinates"

    assert np.all(
        np.less(cuboid_a[:3], cuboid_a[3:])
    ), "cuboid_a must have non-zero volume"

    assert np.all(
        np.less(cuboid_b[:3], cuboid_b[3:])
    ), "cuboid_b must have non-zero volume"

    # Проверка, вписывается ли cuboid_b в cuboid_a
    return np.all(np.less_equal(cuboid_a[:3], cuboid_b[:3])) and np.all(
        np.less_equal(cuboid_b[3:], cuboid_a[3:])
    )


if __name__ == "__main__":
    pass
