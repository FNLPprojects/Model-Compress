"""Utility function for creating the specified learner."""

import tensorflow as tf

from learners.full_precision.learner import FullPrecLearner
from learners.discr_channel_pruning.learner import DisChnPrunedLearner

FLAGS = tf.app.flags.FLAGS

def create_learner(sm_writer, model_helper):
  """Create the learner as specified by FLAGS.learner.

  Args:
  * sm_writer: TensorFlow's summary writer
  * model_helper: model helper with definitions of model & dataset

  Returns:
  * learner: the specified learner
  """

  learner = None
  if FLAGS.learner == 'full-prec':
    learner = FullPrecLearner(sm_writer, model_helper)
  elif FLAGS.learner == 'dis-chn-pruned':
    learner = DisChnPrunedLearner(sm_writer, model_helper)
  else:
    raise ValueError('unrecognized learner\'s name: ' + FLAGS.learner)

  return learner
