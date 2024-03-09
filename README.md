# Garpix-practice
# Классификация наборов грузов для укладки в грузовой контейнер
Contains study project based on https://github.com/luisgarciar/3D-bin-packing

![random_rollout2](https://github.com/Senich17/Garpix-practice/assets/131812061/acc0087e-1881-46de-982b-2c9048820206)

Данный проект представляет собой решение задачи для преддипломной практики, в которой необходимо 
классифицировать наборы грузов с точки зрения того, насколько плотно их удастся уложить в грузовой контейнер.
Разработать модель (решить задачу классификации), которая смогла бы с некоторой известной точностью предсказывать: будет ли для некоторого заданного набора коробок метрика "плотность укладки" больше или равна некоторому заданному значению.

### Основные задачи:

• Распарсить (из json) и проанализировать данные
• Выработать на основе первичного анализа гипотезу о методе решения
• Реализовать выбранные метод (возможно, ML или искусственная нейронная сеть), в том числе "обучить модель" (если потребуется), написать код для взаимодействия с моделью
• Оценить качество результата
• Минимально задокументировать

### Данные:

Известны размеры разнородных коробок ("длина", "ширина",
"высота"), которые собраны в наборы (в наборе коробки разных видов). О нескольких тысячах таких наборов известно насколько плотно их удалось поместить в контейнере при погрузке (известно число - "метрика плотность укладки").
Есть датасет, содержащий сведения о наборах грузов и результатах их укладки (с точки зрения плотности размещения) в ограниченный объем.

# Решение

## Работа с данными

### Предобработка

### Анализ

## Гипотеза

## Реализация

### Настройка гиперпараметров

### Обучение

### Инференс

## Оценка

## Проблемы

## Версии библиотек и окружение

Окружение

Окружение Gym имплементировано в модуле src/packing_env.py

Механизм упаковки

Модуль packaging_engineer (расположенный в src/packing_engine.py) реализует объекты Container и Box, которые используются в среде Gym. Чтобы добавить пользовательские функции (например, разрешить вращение), обратитесь к документации этого модуля.

# Документация

Документация по данному проекту расположена в файле doc.md, с полным описанием функций, а также вознаграждений, которые будут использованы для обучения RL.
