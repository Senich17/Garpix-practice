""" Основной агент для решения задач трехмерной упаковки."""

from typing import Dict


def rnd_agent(observation: Dict) -> Dict:
    """Рандомный агент для среды упаковки.

    Аргументы:
        observation (dict): Наблюдение за средой.

    Возвращается:
        action (dict): Действие, которое необходимо предпринять.
    """
    action = {"position": [0, 0], "box_index": 0}
    return action
