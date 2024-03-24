import tensorflow as tf

class ListContainer(tf.Module):
    def __init__(self, episodes):
        self.episodes = episodes