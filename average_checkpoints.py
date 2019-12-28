# coding=utf-8
"""Checkpoint averaging script."""

import argparse
import os

import numpy as np
import six
import tensorflow as tf


def _variable_is_trainable(name, value):
    _ = name
    # Assume that int variables are not trainable.
    return value.dtype not in (np.int32, np.int64)


def get_checkpoint_variables(checkpoint_path):
    """Returns variables included in a checkpoint.

    Args:
      checkpoint_path: Path to the checkpoint.

    Returns:
      A dictionary mapping variables name to value.
    """
    print(checkpoint_path)
    reader = tf.train.load_checkpoint(checkpoint_path)

    return {
        name: reader.get_tensor(name)
        for name in six.iterkeys(reader.get_variable_to_shape_map())}


def _create_checkpoint_from_variables(variables, output_dir, latest_step=None, session_config=None):
    # The straightforward approach would be to create new variables using a
    # constant_initializer. However, this would save the constant value in the
    # checkpoint meta file which would increase its size dramatically. Instead, we
    # create variables with their default initializer but run an assignment op
    # that writes the new value. Inspired by:
    # github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/bin/t2t_avg_all.py
    if "global_step" in variables:
        latest_step = variables["global_step"]
        del variables["global_step"]
    with tf.Graph().as_default():
        tf_vars = [
            tf.get_variable(
                name,
                shape=value.shape,
                dtype=tf.as_dtype(value.dtype),
                trainable=_variable_is_trainable(name, value))
            for name, value in six.iteritems(variables)]
        placeholders = [tf.placeholder(v.dtype, shape=v.shape)
                        for v in tf_vars]
        assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]

        out_base_file = os.path.join(output_dir, "model.ckpt")
        global_step = tf.get_variable(
            "global_step",
            initializer=tf.constant(latest_step, dtype=tf.int64),
            trainable=False)
        saver = tf.train.Saver(tf.global_variables(), save_relative_paths=True)

        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())
            for p, assign_op, value in zip(placeholders, assign_ops, six.itervalues(variables)):
                sess.run(assign_op, {p: value})
            tf.logging.info("Saving new checkpoint to %s" % output_dir)
            saver.save(sess, out_base_file, global_step=global_step)

    return output_dir


def average_checkpoints(model_dir, output_dir, max_count=8, session_config=None):
    """Averages checkpoints.

    Args:
      model_dir: The directory containing checkpoints.
      output_dir: The directory that will contain the averaged checkpoint.
      max_count: The maximum number of checkpoints to average.
      session_config: Configuration to use when creating the session.

    Returns:
      The path to the directory containing the averaged checkpoint.

    Raises:
      ValueError: if :obj:`output_dir` is the same as :obj:`model_dir`.
    """
    if model_dir == output_dir:
        raise ValueError("Model and output directory must be different")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    checkpoints_path = tf.train.get_checkpoint_state(
        model_dir).all_model_checkpoint_paths
    print(checkpoints_path)
    if len(checkpoints_path) > max_count:
        checkpoints_path = checkpoints_path[-max_count:]
    num_checkpoints = len(checkpoints_path)

    tf.logging.info("Averaging %d checkpoints..." % num_checkpoints)
    tf.logging.info("Listing variables...")

    new_variables = {}
    for i, checkpoint_path in enumerate(checkpoints_path):
        tf.logging.info("Loading checkpoint %s" % checkpoint_path)
        variables = get_checkpoint_variables(checkpoint_path)
        for name, value in six.iteritems(variables):
            if _variable_is_trainable(name, value):
                scaled_value = value / num_checkpoints
                if name in new_variables:
                    new_variables[name] += scaled_value
                else:
                    new_variables[name] = scaled_value
            # Take non trainable variables from the last checkpoint.
            elif i + 1 == num_checkpoints:
                new_variables[name] = value

    return _create_checkpoint_from_variables(
        new_variables,
        output_dir,
        session_config=session_config)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_dir", required=True,
                        help="The model directory containing the checkpoints.")
    parser.add_argument("--output_dir", required=True,
                        help="The output directory where the averaged checkpoint will be saved.")
    parser.add_argument("--max_count", type=int, default=20,
                        help="The maximal number of checkpoints to average.")
    args = parser.parse_args()
    average_checkpoints(args.model_dir, args.output_dir,
                        max_count=args.max_count)


if __name__ == "__main__":
    main()
