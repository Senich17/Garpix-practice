import warnings

import gym
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from src.utils import boxes_generator

from plotly_gif import GIF

import io
from PIL import Image

def make_env(
    container_size,
    num_boxes,
    num_visible_boxes=1,
    seed=0,
    render_mode=None,
    random_boxes=False,
    only_terminal_reward=False,
):
    """
    Параметры

    ----------
    container_size: размер контейнера
    num_boxes: количество упаковываемых ящиков 
    num_visible_boxes: количество ящиков, видимых агенту
    seed: начальное значение для RNG
    render_mode: режим рендеринга для среды
    random_boxes: использовать ли случайные коробки или нет
    only_terminal_reward: использовать ли только конечное вознаграждение или нет
    """
    env = gym.make(
        "PackingEnv-v0",
        container_size=container_size,
        box_sizes=boxes_generator(container_size, num_boxes, seed),
        num_visible_boxes=num_visible_boxes,
        render_mode=render_mode,
        random_boxes=random_boxes,
        only_terminal_reward=only_terminal_reward,
    )
    return env


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    container_size = [10, 10, 10]
    box_sizes2 = [[3, 3, 3], [3, 2, 3], [3, 4, 2], [3, 2, 4], [3, 2, 3]]

    orig_env = gym.make(
        "PackingEnv-v0",
        container_size=container_size,
        box_sizes=box_sizes2,
        num_visible_boxes=1,
        render_mode="human",
        random_boxes=False,
        only_terminal_reward=False,
    )

    env = gym.make(
        "PackingEnv-v0",
        container_size=container_size,
        box_sizes=box_sizes2,
        num_visible_boxes=1,
        render_mode="human",
        random_boxes=False,
        only_terminal_reward=False,
    )

    check_env(env, warn=True)

    model = MaskablePPO("MultiInputPolicy", env, verbose=1)
    print("begin training")
    model.learn(total_timesteps=10)
    print("done training")
    model.save("ppo_mask")

    obs = orig_env.reset()
    done = False
    gif = GIF(gif_name="trained_5boxes.gif", gif_path="../gifs")
    figs = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = orig_env.step(action)
        fig = env.render(mode="human")
        fig_png = fig.to_image(format="png")
        buf = io.BytesIO(fig_png)
        img = Image.open(buf)
        figs.append(img)
    print("done packing")
    env.close()

## сохраняем гифку
    figs[0].save('../gifs/train_5_boxes.gif', format='GIF',
                   append_images=figs[1:],
                   save_all=True,
                   duration=300, loop=0)

    # gif.create_gif(length=5000)
