"""Механизм упаковки: Базовые классы для задачи упаковки
Мы следуем пространственному представлению, показанному ниже, все координаты и длины коробок и контейнеров являются целыми числами.

    x: глубина
    y: длина
    z: высота

       Z
       |
       |
       |________Y
      /
     /
    X

    классы:
        коробка
        контейнер

"""
from copy import deepcopy
from typing import List, Type

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from nptyping import NDArray, Int, Shape

from src.utils import (
    generate_vertices,
    boxes_generator,
    cuboids_intersection,
    cuboid_fits,
)


class Box:
    """Класс трехмерной коробки

    Атрибуты
    ----------
     id_: int
          идентификатор коробки
     position: int
          Координаты положения самого нижнего левого глубокого угла коробки
     size: int
          длины краев коробки
    """

    def __init__(self, size: List[int], position: List[int], id_: int) -> None:
        """Инициализация объекта box

        Параметры
        ----------
        size: List[int]
            Длины краев прямоугольника в порядке (x, y, z) = (глубина, длина, высота)
        position: List[int]
            Координаты положения самого нижнего левого и самого глубокого угла коробки
        id_: int
            идентификатор коробки

        Возвращается
        -------
        Коробочный объект))
        """
        assert len(size) == len(
            position
        ), "Lengths of box size and position do not match"
        assert len(size) == 3, "Box size must be a list of 3 integers"

        assert (
            size[0] > 0 and size[1] > 0 and size[2] > 0
        ), "Lengths of edges must be positive"
        assert (position[0] == -1 and position[1] == -1 and position[2] == -1) or (
            position[0] >= 0 and position[1] >= 0 and position[2] >= 0
        ), "Position is not valid"

        self.id_ = id_
        self.position = np.asarray(position)
        self.size = np.asarray(size)

    def rotate(self, rotation: int) -> None:
        """Вращение коробки

        Параметры
        ----------
        rotation: int
        """
        pass  # потом добавим

    @property
    def area_bottom(self) -> int:
        """Площадь нижней поверхности коробки"""
        return self.size[0] * self.size[1]

    @property
    def volume(self) -> int:
        """Объем коробки"""
        return self.size[0] * self.size[1] * self.size[2]

    @property
    def vertices(self) -> NDArray:
        """Возвращает список с вершинами коробки"""
        vert = generate_vertices(self.size, self.position)
        return np.asarray(vert, dtype=np.int32)

    def __repr__(self):
        return (
            f"Box id: {self.id_}, Size: {self.size[0]} x {self.size[1]} x {self.size[2]}, "
            f"Position: ({self.position[0]}, {self.position[1]}, {self.position[2]})"
        )

    def plot(self, color, figure: Type[go.Figure] = None) -> Type[go.Figure]:
        """Визуализирует коробки

         Параметры
         ----------
        figure: go.Figure
             уизуализация, отображающая коробку

         Возвращается
         -------
         go.Figure
        """
        # Сгенерируем координаты вершин
        vertices = generate_vertices(self.size, self.position).T
        x, y, z = vertices[0, :], vertices[1, :], vertices[2, :]
        # Массивы i, j, k содержат индексы треугольников, которые будут построены (по два на каждую грань прямоугольника).
        # Треугольники имеют вершины (x[i[индекс]], y[j[индекс]], z[k[индекс]]), индекс = 0,1,..7.
        i = [1, 2, 5, 6, 1, 4, 3, 6, 1, 7, 0, 6]
        j = [0, 3, 4, 7, 0, 5, 2, 7, 3, 5, 2, 4]
        k = [2, 1, 6, 5, 4, 1, 6, 3, 7, 1, 6, 0]

        edge_pairs = [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ]
        for (m, n) in edge_pairs:
            vert_x = np.array([x[m], x[n]])
            vert_y = np.array([y[m], y[n]])
            vert_z = np.array([z[m], z[n]])

        if figure is None:
            # визуализация сторон коробки
            figure = go.Figure(
                data=[
                    go.Mesh3d(
                        x=x,
                        y=y,
                        z=z,
                        i=i,
                        j=j,
                        k=k,
                        opacity=1,
                        color=color,
                        flatshading=True,
                    )
                ]
            )
            # Визуализация вершин
            figure.add_trace(
                go.Scatter3d(
                    x=vert_x,
                    y=vert_y,
                    z=vert_z,
                    mode="lines",
                    line=dict(color="black", width=0),
                )
            )

        else:
            # грани
            figure.add_trace(
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    opacity=1,
                    color=color,
                    flatshading=True,
                )
            )
            # Углы
            figure.add_trace(
                go.Scatter3d(
                    x=vert_x,
                    y=vert_y,
                    z=vert_z,
                    mode="lines",
                    line=dict(color="black", width=0),
                )
            )

        return figure


class Container:
    """Класс 3D-контейнера

    Атрибуты
    ----------
    id_: int
         идентификатор контейнера
    size: NDArray[Shape["1,3"],Int]
        Длины граней контейнера
    position: NDArray[Shape["1,3"],Int]
        Координаты самого нижнего левого глубокого угла контейнера
    Box: list[Box]
        Список с коробками, размещенными внутри контейнера
    height_map: NDArray[Shape["*,*"],Int]
        Массив размера (size[0],size[1]), представляющий карту высот (вид сверху) контейнера,
        где height_map[i,j] - текущая высота сложенных элементов в позиции (i,j).
    """

    def __init__(
        self,
        size: NDArray[Shape["1,3"], Int],
        position: NDArray[Shape["1,3"], Int] = None,
        id_: int = 0,
    ) -> None:
        """Инициализация 3D-контейнера

        Параметры
        ----------
        id_: int, опционально
            id контейнера (по умолчанию = 0)
        positions: int, опционально
            Координаты самого нижнего левого глубокого угла контейнера (по усолчанию = 0,0,0)
        size: int
            Длины граней контейнера
        """

        if position is None:
            position = np.zeros(shape=3, dtype=np.int32)

        assert len(size) == len(position), "Sizes of size and position do not match"
        assert len(size) == 3, "Size of size is different from 3"
        position = np.asarray(position)
        np.testing.assert_equal(position[2], 0), "Position is not valid"

        self.id_ = id_
        self.position = np.asarray(position, dtype=np.int32)
        self.size = np.asarray(size, dtype=np.int32)
        self.boxes = []
        self.height_map = np.zeros(shape=(size[0], size[1]), dtype=np.int32)

    @property
    def vertices(self):
        """Возвращаем список вершин контейнера"""
        return generate_vertices(self.size, self.position)

    @property
    def volume(self) -> int:
        """Объем коробок"""
        return self.size[0] * self.size[1] * self.size[2]

    def reset(self):
        """Сброс контейнера в состояние пустого"""
        self.boxes = []
        self.height_map = np.zeros(shape=[self.size[0], self.size[1]], dtype=np.int32)

    def _update_height_map(self, box):
        """Обновляет карту высот после размещения коробки
         Параметры
        ----------
        box: Box
             Коробка, помещаемая внутрь контейнера
        """
        # Добавление высоты новой коробки в координатах x-y, занимаемых этой коробкой
        self.height_map[
            box.position[0] : box.position[0] + box.size[0],
            box.position[1] : box.position[1] + box.size[1],
        ] += box.size[2]

    def __repr__(self):
        return (
            f"Container id: {self.id_}, Size: {self.size[0]} x {self.size[1]} x {self.size[2]}, "
            f"Position: ({self.position[0]}, {self.position[1]}, {self.position[2]})"
        )

    def get_height_map(self):
        """возвращаем карту высот"""
        return deepcopy(self.height_map)

    def check_valid_box_placement(
        self, box: Box, new_pos: NDArray, check_area: int = 100
    ) -> int:
        """
        Параметры
        ----------
        box: Box
            Коробка для размещения
        new_pos: NDArray[int]
            Координаты новой позиции
        check_area: int, по умолчанию = 100
             Процент площади нижней части поля, которая должна поддерживаться в новом положении

        Возвращается
        -------
        int
        """
        assert len(new_pos) == 2

        # Сгенерируйте вершины нижней грани коробки
        v = generate_vertices(np.asarray(box.size), np.asarray([*new_pos, 1]))
        # нижние вершины коробки
        v0, v1, v2, v3 = v[0, :], v[1, :], v[2, :], v[3, :]

        # Сгенерируйте вершины нижней грани контейнера
        w = generate_vertices(self.size, self.position)
        # нижние вершины контейнера
        w0, w1, w2, w3 = w[0, :], w[1, :], w[2, :], w[3, :]

        # Проверка, что коробка находится именно в рамках контейнера
        cond_0 = np.all(np.logical_and(v0[0:2] >= w0[0:2], v0[0:2] <= w3[0:2]))
        cond_1 = np.all(np.logical_and(v1[0:2] >= w0[0:2], v1[0:2] <= w3[0:2]))
        cond_2 = np.all(np.logical_and(v2[0:2] >= w0[0:2], v2[0:2] <= w3[0:2]))
        cond_3 = np.all(np.logical_and(v3[0:2] >= w0[0:2], v3[0:2] <= w3[0:2]))

        if not np.all([cond_0, cond_1, cond_2, cond_3]):
            return 0

        # проверка, что нижние вершины коробки в новом положении находятся на одном уровне
        corners_levels = [
            self.height_map[v0[0], v0[1]],
            self.height_map[v1[0] - 1, v1[1]],
            self.height_map[v2[0], v2[1] - 1],
            self.height_map[v3[0] - 1, v3[1] - 1],
        ]

        if corners_levels.count(corners_levels[0]) != len(corners_levels):
            return 0

        # lev - это уровень (высота), на котором будут расположены нижние углы коробки
        lev = corners_levels[0]
        # bottom_face_lev содержит уровни всех точек на нижней грани
        bottom_face_lev = self.height_map[
            v0[0] : v0[0] + box.size[0], v0[1] : v0[1] + box.size[1]
        ]

        # Проверка, что уровень углов является максимальным из всех точек на нижней грани
        if not np.array_equal(lev, np.amax(bottom_face_lev)):
            return 0

        # Подсчет, сколько точек на нижней грани поддерживаются на высоте, равной lev
        count_level = np.count_nonzero(bottom_face_lev == lev)
        # Проверка процента поддерживаемой площади дна коробки (на высоте, равной lev).
        support_perc = int((count_level / (box.size[0] * box.size[1])) * 100)
        if support_perc < check_area:
            return 0

        dummy_box = deepcopy(box)
        dummy_box.position = [*new_pos, lev]

        # Проверка, что коробка помещается в контейнер на новой позиции
        dummy_box_min_max = [
            dummy_box.position[0],
            dummy_box.position[1],
            dummy_box.position[2],
            dummy_box.position[0] + dummy_box.size[0],
            dummy_box.position[1] + dummy_box.size[1],
            dummy_box.position[2] + dummy_box.size[2],
        ]

        container_min_max = [
            self.position[0],
            self.position[1],
            self.position[2],
            self.position[0] + self.size[0],
            self.position[1] + self.size[1],
            self.position[2] + self.size[2],
        ]

        if not cuboid_fits(container_min_max, dummy_box_min_max):
            return 0

        # Проверка, что нет наложения коробок друг в друга в контейнере
        for other_box in self.boxes:
            if other_box.id_ == dummy_box.id_:
                continue
            other_box_min_max = [
                other_box.position[0],
                other_box.position[1],
                other_box.position[2],
                other_box.position[0] + other_box.size[0],
                other_box.position[1] + other_box.size[1],
                other_box.position[2] + other_box.size[2],
            ]

            if cuboids_intersection(dummy_box_min_max, other_box_min_max):
                return 0

        # если все условия выполнены, позиция является действительной
        return 1

    def action_mask(
        self, box: Box, check_area: int = 100
    ) -> NDArray[Shape["*, *"], Int]:
        """Возвращает массив со всеми возможными позициями для коробки в контейнере
        array[i,j] = 1, если коробка может быть помещена в позицию (i,j), 0 в противном случае

           Параметры
           ----------
           box: Box
               коробка для упаковки
           check_area: int, по умолчанию = 100
                Процент площади нижней части коробки, которая должна поддерживаться в новом положении

           Возвращается
           -------
           np.array(np.int8)
        """

        action_mask = np.zeros(shape=[self.size[0], self.size[1]], dtype=np.int8)
        # Генерируем все возможные положения коробки в контейнере
        for i in range(0, self.size[0]):
            for j in range(0, self.size[1]):
                if (
                    self.check_valid_box_placement(
                        box, np.array([i, j], dtype=np.int32), check_area
                    )
                    == 1
                ):
                    action_mask[i, j] = 1
        return action_mask

    def place_box(self, box: Box, new_position: List[int], check_area=100) -> None:
        """Помещает коробку в контейнер
        Параметры
        ----------
        box: Box
            Поле для размещения
        new_position: List[int]
            Координаты новой позиции
        check_area:
        """
        assert (
            self.check_valid_box_placement(box, new_position, check_area) == 1
        ), "Invalid position for box"
        # проверить height_map, чтобы определить высоту, на которой будет размещена коробка
        height = self.height_map[new_position[0], new_position[1]]
        # Обновление позиции коробки
        box.position = np.asarray([*new_position, height], dtype=np.int32)
        # Добавляем коробку в контейнеор
        self.boxes.append(box)
        # Обновляем height_map
        self._update_height_map(box)

    def plot(self, figure: Type[go.Figure] = None) -> Type[go.Figure]:
        """Добавляет визуализацию

        Параметры
        ----------
        figure: go.Figure, по умолчанию = None
         Поле где должна быть отрисована коробка
        Возвращается
        -------
            go.Figure
        """
        if figure is None:
            figure = go.Figure()

        # Генерируем все вершины и пары ребер, нумерация которых объяснена в функции utils.generate_vertices
        vertices = generate_vertices(self.size, self.position).T
        x, y, z = vertices[0, :], vertices[1, :], vertices[2, :]
        edge_pairs = [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ]

        # Добавляем линию между каждой парой ребер
        for (m, n) in edge_pairs:
            vert_x = np.array([x[m], x[n]])
            vert_y = np.array([y[m], y[n]])
            vert_z = np.array([z[m], z[n]])
            figure.add_trace(
                go.Scatter3d(
                    x=vert_x,
                    y=vert_y,
                    z=vert_z,
                    mode="lines",
                    line=dict(color="yellow", width=3),
                )
            )

        color_list = px.colors.qualitative.Dark24

        for item in self.boxes:
            # item_color = color_list[-2]
            item_color = color_list[(item.volume + item.id_) % len(color_list)]
            figure = item.plot(item_color, figure)

        # Выбор угла для визуализации
        # camera = dict(eye=dict(x=2, y=2, z=0.1))

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25),
        )

        
        figure.update_layout(
            showlegend=False,
            scene_camera=camera,
            width=1200,
            height=1200,
            template="plotly_dark",
        )

        max_x = self.position[0] + self.size[0]
        max_y = self.position[1] + self.size[1]
        max_z = self.position[2] + self.size[2]
        figure.update_layout(
            scene=dict(
                xaxis=dict(nticks=int(max_x + 2), range=[0, max_x + 5]),
                yaxis=dict(nticks=int(max_y + 2), range=[0, max_y + 5]),
                zaxis=dict(nticks=int(max_z + 2), range=[0, max_z + 5]),
                aspectmode="cube",
            ),
            width=1200,
            margin=dict(r=20, l=10, b=10, t=10),
        )

        figure.update_scenes(
            xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False
        )
        figure.update_scenes(
            xaxis_showticklabels=False,
            yaxis_showticklabels=False,
            zaxis_showticklabels=False,
        )

        return figure

    def first_fit_decreasing(self, boxes: List[Box], check_area: int = 100) -> None:
        """Помещает все ящики в контейнер, используя параметры эвристического метода уменьшения первого соответствия

        ----------
        boxes: List[Box]
            Список ящиков для размещения
        check_area: int, по умолчанию = 100
            Процент площади нижней части ящика, которая должна поддерживаться в новом положении
        """
        # Сортировка коробок в порядке убывания их объема
        boxes.sort(key=lambda x: x.volume, reverse=True)

        for box in boxes:
            # Найти позицию для размещения коробки
            action_mask = self.action_mask(box, check_area)

            # top lev максимальный уровень, где возможно размещение коробки
            # в соответствии с ее высотой
            top_lev = self.size[2] - box.size[2]
            # max_occupied максимальныя высота, которую занимает коробка
            max_occupied = np.max(self.height_map)
            lev = min(top_lev, max_occupied)

            # Находим первую позицию, куда можно поместить коробку, начиная с
            # верхнего уровня и спускаясь вниз
            k = lev
            while k >= 0:
                locations = np.zeros(shape=(self.size[0], self.size[1]), dtype=np.int32)
                kth_level = np.logical_and(
                    self.height_map == k, np.equal(action_mask, 1)
                )
                if kth_level.any():
                    locations[kth_level] = 1
                    # Находим первую позицию для размещения коробки
                    position = [
                        np.nonzero(locations == 1)[0][0],
                        np.nonzero(locations == 1)[1][0],
                    ]
                    # Размещение коробки на первой найденной позиции
                    self.place_box(box, position, check_area)
                    break
                k -= 1


if __name__ == "__main__":
    len_bin_edges = [10, 10, 10]
    # Сгенерированные коробки точно поместятся в контейнер размера [10,10,10]
    boxes_sizes = boxes_generator(len_bin_edges, num_items=64, seed=42)
    boxes = [
        Box(size, position=[-1, -1, -1], id_=i) for i, size in enumerate(boxes_sizes)
    ]
    # Мы упаковываем коробки в контейнер большего размера, поскольку эвристическое правило не является оптимальным
    container = Container(np.array([12, 12, 12], dtype=np.int32))
    # Параметр 'check_area' указывает процент от нижней области поля, который должен поддерживаться
    container.first_fit_decreasing(boxes, check_area=100)
    # визуалищируем
    fig = container.plot()
    fig.show()
