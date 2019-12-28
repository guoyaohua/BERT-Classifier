# coding=utf-8
"""Model saver."""
import glob
import os

import tensorflow as tf


class ModelSaver:
    def __init__(self, keep_checkpoint_max=10):
        self.keep_checkpoint_max = keep_checkpoint_max
        self.checkpoint_dict = {}
        self.saver = tf.train.Saver(max_to_keep=keep_checkpoint_max)

    def check_and_save_model(self, save_path, metric, sess):
        print(self.checkpoint_dict)
        if len(self.checkpoint_dict) < self.keep_checkpoint_max:
            save_path = os.path.join(save_path, "model-%f.ckpt" % metric)
            self.saver.save(sess, save_path)
            self.checkpoint_dict[metric] = save_path
        else:
            min_key = None
            for k in self.checkpoint_dict.keys():
                if min_key == None:
                    min_key = k
                else:
                    min_key = k if min_key > k else min_key
            if metric > min_key:
                ckpt_list = glob.glob(self.checkpoint_dict[min_key]+"*")
                for path in ckpt_list:
                    os.remove(path)
                self.checkpoint_dict.pop(min_key)
                # update saver
                self.saver.set_last_checkpoints(
                    list(self.checkpoint_dict.values()))
                save_path = os.path.join(save_path, "model-%f.ckpt" % metric)
                self.saver.save(sess, save_path)
                self.checkpoint_dict[metric] = save_path
            else:
                tf.logging.info(
                    "Skip to save this ckpt, metric of this time is %f smaller than the min metric %f which had been saved." % (metric, min_key))
                return
        tf.logging.info("Successfully save checkpoints")
