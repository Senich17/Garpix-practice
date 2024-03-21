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

Определение задачи и допущения:

Среда состоит из списка трехмерных коробок разного размера и одного контейнера фиксированного размера. Цель состоит в том, чтобы упаковать как можно больше коробок в контейнер, минимизируя пустой объем. Мы предполагаем, что вращение коробок невозможно.

## Работа с данными

### Предобработка

Данные были распарсены и предобработаны следующим образом: в файле data_cargo_for_model1.csv находятся данные о коробках, которые были у заказчика.

Дополнительно создан файл с параметрами грузового пространства заказчика (data_cargo_space_full.csv).

### Анализ



## Гипотеза

Гипотеза: Нейронные модели обучения с подкреплением хорошо подходят для решения проблемы рюкзака из-за их способности изучать оптимальную "политику" принятия решений в динамичных и неопределенных средах, характеристики которых тесно связаны с природой проблемы рюкзака.

Почему мы так считаем:

1. Динамическая среда: Задача с рюкзаком включает в себя выбор подмножества предметов для максимизации общей вместимости при соблюдении ограничений этой же вметимости рюкзака. Эта задача по своей сути динамична, поскольку выбор каждого элемента влияет на доступное пространство и потенциальные будущие варианты. Модели RL преуспевают в динамических средах, итеративно обучаясь принимать решения на основе обратной связи, полученной из окружающей среды.

2. Неопределенная среда: Неопределенность в проблеме рюкзака возникает из-за огромного количества возможных комбинаций предметов и связанных с ними значений и весов. Модели RL способны справляться с такой неопределенностью, исследуя различные действия и делая "выводы" из результатов, постепенно приближаясь к оптимальным решениям.

3. Оптимизация политики: модели RL нацелены на максимизацию кумулятивного вознаграждения с течением времени путем изучения оптимальных политик принятия решений. В контексте проблемы с рюкзаком вознаграждение может быть определено как общий объем выбранных предметов. Алгоритмы RL, такие как Q-learning или Deep Q-Networks (DQN), могут изучать политики, которые эффективно балансируют между максимизацией общей вместимости и соблюдением ограничений пропускной способности.

4. Гибкость и адаптируемость: Модели RL являются гибкими и адаптируемыми, что позволяет им учитывать различные требования к решению проблем и ограничения. Эта гибкость важна в задаче с рюкзаком, где различные сценарии могут потребовать корректировки критериев принятия решений на основе таких факторов, как количество и объем предметов, вес и вместимость рюкзака.

5. Способность обрабатывать большие пространства состояний: Проблема с рюкзаком часто связана с большим пространством состояний, учитывая многочисленные комбинации предметов. Алгоритмы RL, особенно те, которые используют глубокие нейронные сети, хорошо справляются с пространствами состояний высокой размерности путем аппроксимации функций значений или политик, обеспечивая эффективное исследование и эксплуатацию пространства решений.

## Реализация

Функция boxes_generator в файле utils.py генерирует экземпляры задачи упаковки 3D-контейнеров с использованием алгоритма, описанного в разделе Ranked Reward: [Включение обучения с подкреплением при самостоятельной игре для комбинаторной оптимизации](https://arxiv.org/pdf/1807.01672.pdf) (алгоритм 2, приложение).

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

Чтобы запустить код, необходимо установить зависимосьти

cd 3D-bin-packing
pip install -r requirements.txt

# Документация

Документация по данному проекту расположена в файле doc.md, с полным описанием функций, а также вознаграждений, которые будут использованы для обучения RL.
