import numpy as np
import tensorflow_datasets as tfds
import pickle
import os
import argparse
import tensorflow as tf
from PIL import Image
import random
import matplotlib.pyplot as plt

DATA_PATH = '/home/dvenuto/af_dt/af_dt/language-table/data'
FRAMES_DESTINATION_DIR_PATH = '/home/dvenuto/af_dt/af_dt/language-table/language_table/start_end_frames'

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
      destination_dir_path = os.path.join(DATA_PATH, instruction_connected_by_underscores, str(episode_count), str(index))
      os.makedirs(destination_dir_path, exist_ok=False)
      tf.data.experimental.save(episode['steps'], destination_dir_path)

def create_and_save_fig(rbg_array, instruction, file_ending):
    frame_img_save_path = os.path.join(FRAMES_DESTINATION_DIR_PATH, f"{instruction}-{file_ending}.png")
    plt.imshow(rbg_array)
    plt.title(instruction)
    plt.savefig(frame_img_save_path)
    return frame_img_save_path

#TODO: Just perform a sanity check to ensure that all steps in all episodes
# actually do have the same language goal.
def get_start_and_end_frames(target_start_and_end_frames_count, target_instruction):
  episodes = []
  try:
    episode_count = 0
    total_number_of_steps = 0
    total_starts_and_ends = 0
    print("Go 1")
    for episode in get_episode_ds(dataset_paths['language_table']):

        first_step = next(iter(episode['steps'].as_numpy_iterator()))
        instruction = decode_inst(first_step['observation']['instruction'])

        if instruction.strip() == target_instruction.strip():

            episodes.append(episode)
            start_done = False
            end_done = False
            random_file_ending = random.randint(0, 2000000)

            for step in episode['steps'].as_numpy_iterator():
               if start_done and end_done:
                  total_starts_and_ends += 1
                  break
               rgb_array = step['observation']['rgb']
               if step['is_first']:
                  create_and_save_fig(rgb_array, instruction, str(random_file_ending) + "start")
                  start_done = True               
               if step['is_last']:
                  create_and_save_fig(rgb_array, instruction, str(random_file_ending) + "end")
                  end_done = True

            if total_starts_and_ends == target_start_and_end_frames_count:
               break

        episode_count += 1

    print(f"Total number of steps is: {total_number_of_steps}")
  except Exception as e:
     print(e)
     print("No more records left to process. Need to download more.")
  print(f"The episode count fell short by: {target_start_and_end_frames_count - len(episodes)}") 
  print(f"The total number of episodes in the episode_ds is {episode_count}")

if __name__=='__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--target_episode_count', default=2, help="Number of episodes wanted.")
   parser.add_argument('--target_instruction', default='slide the green circle into the top side of the yellow hexagon')
   args = parser.parse_args()
   get_start_and_end_frames(args.target_episode_count, args.target_instruction)

        # `break`point()
        # set_of_instructions = set()
        # for step in episode['steps'].as_numpy_iterator():
        #   instruction = decode_inst(step['observation']['instruction'])
        #   set_of_instructions.add(instruction)
        #   total_number_of_steps += 1
        #   print(total_number_of_steps)
        # print(f"Number of instructions in episode {episode_count}: {len(set_of_instructions)}")