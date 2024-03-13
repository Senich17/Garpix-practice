"""
Packing Gym: Открытая среда для решения задач трехмерной упаковки.
Мы следуем пространственному представлению, показанному ниже, все координаты и длины коробок и контейнеров являются целыми числами.

    x: глубина
    y: длинна
    z: высота

       Z
       |
       |
       |________Y
      /
     /
    X

    Классы:
        Коробка
        Контейнер

"""
import copy
from typing import List, Tuple, Union

import gym
import numpy as np
import plotly.graph_objects as go
from gym.spaces import Discrete, MultiDiscrete
from gym.utils import seeding
from nptyping import NDArray

from src.packing_kernel import Box, Container


class PackingEnv(gym.Env):
    """Класс представляющий среду упаковки.

    Описание:
        Среда состоит из 3D-контейнера и начального списка 3D-ящиков, цель
        состоит в том, чтобы упаковать коробки в контейнер, минимизируя пустое пространство. Мы предполагаем
        что контейнер загружается сверху.

        Состояние контейнера представлено двумерным массивом, хранящим карту высот (вид сверху)
        контейнера и список размеров предстоящих коробок.
        
        Действие:
        Тип: Дискретный(container.size[0]*container.size[1]*num_visible_boxes)
        Агент выбирает целое число j в диапазоне [0, container.size[0]*container.size[1]*num_visible_boxes)),
        и действие интерпретируется следующим образом: поле с индексом j // (container.size[0]*container.size[1])
        помещается в положение (x,y) = (j//container.size[1], j%container.size[1]) в контейнере.

        Награда:
        В конце эпохи агенту выдается вознаграждение, равное соотношению между объемом
        из упакованных коробок и объема контейнера.

        Начальное состояние:
        height_map инициализируется как нулевой массив, а список предстоящих блоков инициализируется как случайный список
        num_visible_boxes длины из полного списка блоков.

        Завершение эпохи:
        Эпоха завершается, когда все коробки помещаются в контейнер или когда больше нельзя упаковать коробки
        в контейнер.

    """

    metadata = {"render_modes": ["human", "rgb_array", "None"], "render_fps": 4}

    def __init__(
        self,
        container_size: List[int],
        box_sizes: List[List[int]],
        num_visible_boxes: int = 1,
        render_mode: str = "None",
        options: dict = None,
        random_boxes: bool = False,
        only_terminal_reward: bool = True,
    ) -> None:
        """инициализация среды.

         параметры:
        ----------:
            container_size: размер уонтейнера в виде [lx,ly,lz]
            box_sizes: размеры загружаемых коробок в виде [[lx,ly,lz],...]
            num_visible_boxes: количество видимых коробок для агента
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Этот пункт определяет, генерируются ли поля случайным образом при каждом сбросе среды
        self.random_boxes = random_boxes
        # Этот пункт определяет выдачу награды
        self.only_terminal_reward = only_terminal_reward

        # TO DO: Добавить параметр для проверки плошади коробок
        assert num_visible_boxes <= len(box_sizes)
        self.container = Container(container_size)
        # Первоначальный список всех ящиков, которые должны быть помещены в контейнер.

        self.initial_boxes = [
            Box(box_size, position=[-1, -1, -1], id_=index)
            for index, box_size in enumerate(box_sizes)
        ]

        self.num_initial_boxes = len(self.initial_boxes)

        # Список коробок, которые еще не упакованы и не видны агенту
        self.unpacked_hidden_boxes = self.initial_boxes.copy()
        # Список коробок, которые уже упакованы
        self.packed_boxes = []
        # Список коробок, которые могли бы быть упакованы, но не поместились
        self.skipped_boxes = []

        # Список и количество коробок, которые еще не упакованы и видны агенту
        self.num_visible_boxes = num_visible_boxes
        self.unpacked_visible_boxes = []
        self.state = {}
        self.done = False

        # Массив для определения мультидискретного пространства со списком размеров видимых полей
        # Верхняя граница для записей в мультидискретном пространстве не является включающей - мы добавляем 1 к каждой координате
        box_repr = np.zeros(shape=(num_visible_boxes, 3), dtype=np.int32)
        box_repr[:] = self.container.size + [1, 1, 1]
        # Преобразуйте список размеров видимых блоков в одномерный массив
        box_repr = np.reshape(box_repr, newshape=(num_visible_boxes * 3,))

        # Массив для определения мультидискретного пространства с картой высот контейнера
        height_map_repr = np.ones(
            shape=(container_size[0], container_size[1]), dtype=np.int32
        ) * (container_size[2] + 1)
        # Преобразование карты высот в одномерный массив
        height_map_repr = np.reshape(
            height_map_repr, newshape=(container_size[0] * container_size[1],)
        )

        # Словарь, чтобы определить пространство наблюдения
        observation_dict = {
            "height_map": MultiDiscrete(height_map_repr),
            "visible_box_sizes": MultiDiscrete(box_repr),
        }

        # пространство наблюдения
        self.observation_space = gym.spaces.Dict(observation_dict)
        # пространство действий
        self.action_space = Discrete(
            container_size[0] * container_size[1] * num_visible_boxes
        )

        # задаем начальную action_mask как нулевой массив
        self.action_mask = np.zeros(
            shape=(
                self.container.size[0]
                * self.container.size[1]
                * self.num_visible_boxes,
            ),
            dtype=np.int32,
        )

    def seed(self, seed: int = 42):
        """Запуск генератора случайных чисел для среды.
        Параметры
        -----------
            seed: int
            seed для окружающей среды.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def action_to_position(self, action: int) -> Tuple[int, NDArray]:
        """Преобразует индекс в кортеж с индексом ячейки
        и позицией в контейнере.
        Параметры
        ----------
            action:
            индекс int, подлежащий преобразованию.
        
        Возвращается
        -------
            box_index: внутренний
                индекс коробки, которая будет упакована.
            позиция: ndarray
                Позиция в контейнере.
        """
        box_index = action // (self.container.size[0] * self.container.size[1])
        res = action % (self.container.size[0] * self.container.size[1])

        position = np.array(
            [res // self.container.size[0], res % self.container.size[0]]
        )

        return box_index, position.astype(np.int32)

    def position_to_action(self, position, box_index=0):
        """Преобразует позицию в контейнере в индекс действия
        Возвращается
        -------
            действие:
            индекс в контейнере.
        """
        action = (
            box_index * self.container.size[0] * self.container.size[1]
            + position[0] * self.container.size[0]
            + position[1]
        )
        return action

    def reset(self, seed=None, options=None) -> Tuple:
        """Сброс среды.
        Параметры
        ----------
            seed: int
                Начальное значение для среды.
            options: dict
                Параметры для среды.
        Возвращается
        ----------
            obs, info: кортеж с начальным состоянием и словарь с информацией об окружающей среде.
        """

        self.container.reset()

        if self.random_boxes:
            box_sizes = boxes_generator(
                self.container.size, num_items=self.num_initial_boxes
            )
            self.initial_boxes = [
                Box(box_size, position=[-1, -1, -1], id_=index)
                for index, box_size in enumerate(box_sizes)
            ]

        # Сброс списка коробок, которые еще не упакованы и не видны агенту
        self.unpacked_hidden_boxes = copy.deepcopy(self.initial_boxes)

        # Сброс списока коробок, видимых агенту, и удаление их из списка
        # скрытых не упакованных коробок, подлежащих упаковке
        self.unpacked_visible_boxes = copy.deepcopy(
            self.unpacked_hidden_boxes[0 : self.num_visible_boxes]
        )
        del self.unpacked_hidden_boxes[0 : self.num_visible_boxes]

        # Сброс упакованных коробок
        self.packed_boxes = self.container.boxes

        # Установка списка видимых размеров коробки в области наблюдения
        visible_box_sizes = np.asarray(
            [box.size for box in self.unpacked_visible_boxes]
        )

        # Сброс состояния среды
        hm = np.asarray(self.container.height_map, dtype=np.int32)
        hm = np.reshape(hm, (self.container.size[0] * self.container.size[1],))

        # Установка начальной пустой маски action_mask
        self.action_mask = self.action_masks

        vbs = np.reshape(visible_box_sizes, (self.num_visible_boxes * 3,))
        self.state = {"height_map": hm, "visible_box_sizes": vbs}

        self.done = False
        self.seed(seed)

        return self.state

    def calculate_reward(self, reward_type: str = "terminal_step") -> float:
        """рассчитфвает вознаграждение за совершенное действие.
        Возвращается:
        ----------
            награда: Вознаграждение за совершенное действие.
        """
        # Объем упакованных коробок
        packed_volume = np.sum([box.volume for box in self.packed_boxes])

        if reward_type == "terminal_step":
            # Вознаграждение за последний шаг
            container_volume = self.container.volume
            reward = packed_volume / container_volume
        elif reward_type == "interm_step":
            min_x = min([box.position[0] for box in self.packed_boxes])
            min_y = min([box.position[1] for box in self.packed_boxes])
            min_z = min([box.position[2] for box in self.packed_boxes])
            max_x = max([box.position[0] + box.size[0] for box in self.packed_boxes])
            max_y = max([box.position[1] + box.size[1] for box in self.packed_boxes])
            max_z = max([box.position[2] + box.size[2] for box in self.packed_boxes])

            # Вознаграждение за промежуточный шаг
            reward = packed_volume / (
                (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
            )
        else:
            raise ValueError("Invalid reward type")

        return reward

    def step(self, action: int) -> Tuple[NDArray, float, bool, dict]:
        """Шаг.
        Параметры:
        -----------
            action: целое число с указанием действия, которое необходимо выполнить.
        Возвращается:
        ----------
            observation: Словарь с наблюдением среды.
            reward: Вознаграждение за действие.
            terminated: Завершается ли эпоха.
            info: Словарь с дополнительной информацией.
        """

        # Получаем индекс и положение коробки, которая будет упакована в контейнер
        box_index, position = self.action_to_position(action)
        # если коробка является dummy, пропустить этот шаг
        if box_index >= len(self.unpacked_visible_boxes):
            return self.state, 0, self.done, {}

        # Если коробка не является dummy, проверяем, допустимо ли действие
        # TO DO: добавить область проверки параметров, добавить информацию, вернуть информацию
        if (
            self.container.check_valid_box_placement(
                self.unpacked_visible_boxes[box_index], position, check_area=100
            )
            == 1
        ):
            # Помещаем коробку в контейнер и удаляем ее из списка неупакованных видимых коробок
            if self.num_visible_boxes > 1:
                self.container.place_box(
                    self.unpacked_visible_boxes.pop(box_index), position
                )
            else:
                self.container.place_box(self.unpacked_visible_boxes[0], position)
                self.unpacked_visible_boxes = []
            # Обновление карты высот, изменение ее формы, добавление в пространство наблюдения
            self.state["height_map"] = np.reshape(
                self.container.height_map,
                (self.container.size[0] * self.container.size[1],),
            )
            # Обновления списка упакованных коробок
            self.packed_boxes = self.container.boxes
            # установка вознаграждения
            if self.only_terminal_reward:
                reward = 0
            else:
                reward = self.calculate_reward(reward_type="interm_step")

            # Если действие неприемлимо, отправляем коробку в список пропущенных
        else:
            self.skipped_boxes.append(self.unpacked_visible_boxes.pop(box_index))
            reward = 0

        # Обновление списка неупакованных скрытых коробок
        if len(self.unpacked_hidden_boxes) > 0:
            self.unpacked_visible_boxes.append(self.unpacked_hidden_boxes.pop(0))

        # Если коробок для упаковки не осталось, то завершаем эпоху
        if len(self.unpacked_visible_boxes) == 0:
            self.done = True
            terminated = self.done
            reward = self.calculate_reward(reward_type="terminal_step")
            self.state["visible_box_sizes"] = [[0, 0, 0]] * self.num_visible_boxes
            return self.state, reward, terminated, {}

        if len(self.unpacked_visible_boxes) == self.num_visible_boxes:
            # Обновление списка видымых коробок в пространстве наблюдений
            visible_box_sizes = np.asarray(
                [box.size for box in self.unpacked_visible_boxes]
            )
            self.state["visible_box_sizes"] = np.reshape(
                visible_box_sizes, (self.num_visible_boxes * 3,)
            )
            terminated = False
            self.state
            return self.state, reward, terminated, {}

        if len(self.unpacked_visible_boxes) < self.num_visible_boxes:
            # Если коробок меньше, чем максимальное количество видимых коробок, то добавляем dummy коробки
            dummy_box_size = self.container.size
            num_dummy_boxes = self.num_visible_boxes - len(self.unpacked_visible_boxes)
            box_size_list = [box.size for box in self.unpacked_visible_boxes] + [
                dummy_box_size
            ] * num_dummy_boxes
            visible_box_sizes = np.asarray(box_size_list)
            self.state["visible_box_sizes"] = np.reshape(
                visible_box_sizes, (self.num_visible_boxes * 3,)
            )
            terminated = False
            return self.state, reward, terminated, {}

    # @property
    def action_masks(self) -> List[bool]:
        """Получите маску действия из среды.
          Параметры
        Возвращается
        ----------
            np.ndarray: массив с маской действия."""
        act_mask = np.zeros(
            shape=(
                self.num_visible_boxes,
                self.container.size[0] * self.container.size[1],
            ),
            dtype=np.int8,
        )

        for index in range(len(self.unpacked_visible_boxes)):
            acm = self.container.action_mask(
                box=self.unpacked_visible_boxes[index], check_area=100
            )
            act_mask[index] = np.reshape(
                acm, (self.container.size[0] * self.container.size[1],)
            )
        return [x == 1 for x in act_mask.flatten()]

    def render(self, mode=None) -> Union[go.Figure, NDArray]:

        """Рендер среды.
        Аргументы:
            mode: Режим для рендера.
        """

        if mode is None:
            pass

        elif mode == "human":
            fig = self.container.plot()
            # fig.show()
            return fig
        #
        elif mode == "rgb_array":
            import io
            from PIL import Image

            fig_png = self.container.plot().to_image(format="png")
            buf = io.BytesIO(fig_png)
            img = Image.open(buf)
            return np.asarray(img, dtype=np.int8)
        else:
            raise NotImplementedError

    def close(self) -> None:
        """Закоытие среды."""
        pass


if __name__ == "__main__":
    from src.utils import boxes_generator
    from gym import make
    import warnings
    from plotly_gif import GIF

    # Игнорируем plotly и gym предупреждения
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Инициализация среды
    env = make(
        "PackingEnv-v0",
        container_size=[10, 10, 10],
        box_sizes=boxes_generator([10, 10, 10], 64, 42),
        num_visible_boxes=1,
    )
    obs = env.reset()

    gif = GIF(gif_name="random_rollout.gif", gif_path="../gifs")
    for step_num in range(80):
        fig = env.render()
        gif.create_image(fig)
        action_mask = obs["action_mask"]
        action = env.action_space.sample(mask=action_mask)
        obs, reward, done, info = env.step(action)
        if done:
            break

    gif.create_gif()
    gif.save_gif("random_rollout.gif")
