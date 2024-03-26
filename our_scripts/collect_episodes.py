import numpy as np
import tensorflow_datasets as tfds
import pickle
import os
import argparse
import tensorflow as tf
from random import randint

DATA_PATH = '/home/dvenuto/af_dt/af_dt/language-table/data'

random_number = randint(0, 1000)

dataset_paths = {'language_table': '/home/dvenuto/scratch/data_real_robot_2'}


def decode_inst(inst):
    """Utility to decode encoded language instruction"""
    return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")


def get_episode_ds(data_path):
    builder = tfds.builder_from_directory(data_path)
    episode_ds = builder.as_dataset(split='train')
    return episode_ds


def persist_episodes(episodes: list, instruction_connected_by_underscores: str, episode_count: int):
    for index, episode in enumerate(episodes):
        destination_dir_path = os.path.join(
            DATA_PATH, instruction_connected_by_underscores, str(episode_count), str(index))
        os.makedirs(destination_dir_path, exist_ok=False)
        tf.data.experimental.save(episode['steps'], destination_dir_path)

# TODO: Just perform a sanity check to ensure that all steps in all episodes
# actually do have the same language goal.


def collect_episodes(target_episode_count: int):
    instrut_list = [
        "separate the yellow blocks",
        "separate the green blocks",
        "separate the yellow blocks from each other",
        "push the green circle into the group of blocks",
        "separate the green blocks from each other",
        "separate the green circle from the group of blocks",
        "move the green circle to the center of the board",
        "slightly touch the green circle",
        "push the green circle towards the center of the board",
        "move the green circle towards the center of the board",
        "move your arm to the right of the green circle",
        "move the green circle to the bottom center of the board",
        "push the green circle to the center of the board",
        "move the green circle to the top center of the board",
        "move the arm to the bottom of the green circle",
        "place the arm below the green circle",
        "move your arm towards the green circle",
        "move your arm to the left of the green circle",
        "move your arm below the green circle",
        "move your arm away from the green circle",
        "move the arm away from the green circle",
        "separate the green blocks from the group of blocks",
        "push the green circle towards the top center of the board",
        "place the green circle at the center of the board",
        "move the arm to the left of the green circle",
        "touch the green circle",
        "move the green circle to the center",
        "separate yellow blocks from each other",
        "push the green circle to the top center of the board",
        "move the green circle to the top left of the board"
    ]
    epses_per_instruction = {}
    instruct_eps_count = {}

    for instruc in instrut_list:
        epses_per_instruction[instruc] = []
        instruct_eps_count[instruc] = 0

    try:

        episode_count = 0
        total_number_of_steps = 0
        print("Go 1")

        for episode in get_episode_ds(dataset_paths['language_table']):
            first_step = next(iter(episode['steps'].as_numpy_iterator()))
            instruction = decode_inst(first_step['observation']['instruction'])
            if instruction.strip() in instrut_list:
                epses_per_instruction[instruction].append(episode)
                instruct_eps_count[instruction] += 1
                print(
                    f"Current episode count with target: {instruction} is {len(epses_per_instruction[instruction])}")
            episode_count += 1
        print(f"Total number of steps is: {total_number_of_steps}")

    except Exception as e:
        #     if len(episodes) != 0:
        #          persist_episodes(episodes, os.path.join('_'.join(target_instruction.split(' '))), str(target_episode_count))
        print(e)
        print("No more records left to process. Need to download more.")
#  print(f"The episode count fell short by: {target_episode_count - len(episodes)}")
#  print(f"The total number of episodes in the episode_ds is {episode_count}")
    for instruct in epses_per_instruction:
        print(
            f"The total number episodes for instruction {instruct} is {len(epses_per_instruction[instruct])}")
        persist_episodes(epses_per_instruction[instruct], os.path.join(
            '_'.join(instruct.split(' '))), target_episode_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_episode_count', default=4,
                        help="Number of episodes wanted.")
    parser.add_argument('--target_instruction',
                        default='push the red star to the bottom right of the board')
    args = parser.parse_args()
    collect_episodes(args.target_episode_count)

    # `break`point()
    # set_of_instructions = set()
    # for step in episode['steps'].as_numpy_iterator():
    #   instruction = decode_inst(step['observation']['instruction'])
    #   set_of_instructions.add(instruction)
    #   total_number_of_steps += 1
    #   print(total_number_of_steps)
    # print(f"Number of instructions in episode {episode_count}: {len(set_of_instructions)}")
