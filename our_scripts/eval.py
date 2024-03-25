import numpy as np
import tensorflow_datasets as tfds
import pickle
import os
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from random import randint
from yellow_hex_green_circle import YellowHexGreenCircle

COLLECTED_EPISODES_SOURCE_DIR_PATH = '/home/dvenuto/af_dt/af_dt/language-table/data'
FRAMES_DESTINATION_DIR_PATH = '/home/dvenuto/af_dt/af_dt/language-table/language_table/frames'
RESULTS_DIR = '/home/dvenuto/af_dt/af_dt/language-table/language_table/results'


def decode_inst(inst):
    """Utility to decode encoded language instruction"""
    return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")


def get_episodes(target_episode_count, target_instruction):
    source_dir_path = os.path.join(COLLECTED_EPISODES_SOURCE_DIR_PATH, '_'.join(
        target_instruction.split(' ')), str(target_episode_count))
    if not os.path.exists(source_dir_path):
        print(f"""Either data for {target_instruction} has not been
              persisted or the exact target episode count {target_episode_count}
              has not been persisted.
              """)
    episodes = []
    for episode_num in range(target_episode_count):
        episodes.append(tf.data.experimental.load(
            os.path.join(source_dir_path, str(episode_num))))
    return episodes


def create_and_save_fig(rbg_array, instruction, file_ending):
    frame_img_save_path = os.path.join(
        FRAMES_DESTINATION_DIR_PATH, f"{file_ending}.png")
    plt.imshow(rbg_array.numpy())
    plt.title(decode_inst(instruction.numpy()))
    plt.savefig(frame_img_save_path)
    return frame_img_save_path


def run_loop(episodes):
    checker = YellowHexGreenCircle()
    random_file_ending = randint(1, 123123123123)
    with open(os.path.join(RESULTS_DIR, f"result_{random_file_ending}.txt"), 'w') as f:
        print_statement = f"Random file ending is: {random_file_ending}\n"
        print(print_statement)
        f.write(print_statement)
        for index, episode in enumerate(episodes):
            print_statement = f"For episode: {index + 1}\n"
            print(print_statement)
            f.write(print_statement)
            for step in episode:
                print_statement = f"Reward {step['reward']}\n"
                print(print_statement)
                f.write(print_statement)
                observation = step['observation']
                frame_img_path = create_and_save_fig(
                    observation['rgb'], observation['instruction'], random_file_ending)
                # TODO: Add GPT_CHECKER
                reward_from_checker = checker.reward(frame_img_path)
                print_statement = f"Reward by checker {reward_from_checker}\n"
                print(print_statement)
                f.write(print_statement)
                if step['is_last']:
                    checker.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_episode_count', default=4,
                        help="Number of episodes wanted.")
    parser.add_argument('--target_instruction',
                        default='place the yellow heart right to the green circle')
    args = parser.parse_args()
    episodes = get_episodes(args.target_episode_count, args.target_instruction)
    run_loop(episodes)

    # print(f"Start step: {step['is_first']}")
    # print(f"Last step: {step['is_last']}")

