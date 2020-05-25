"""Abstract class for learners."""

from abc import ABC
from abc import abstractmethod
import os
import shutil
import subprocess
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('summ_step', 100, 'summarizaton step size')
tf.app.flags.DEFINE_integer('save_step', 10000, 'model saving step size')
tf.app.flags.DEFINE_string('save_path', './models/model.ckpt', 'model\'s save path')
tf.app.flags.DEFINE_string('save_path_eval', './models_eval/model.ckpt',
                           'model\'s save path for evaluation')
tf.app.flags.DEFINE_boolean('enbl_warm_start', False, 'enable warm start for training')

class AbstractLearner(ABC):  # pylint: disable=too-many-instance-attributes
  """Abstract class for learners.

  A learner should take a ModelHelper object as input, which includes the data input pipeline and
    model definition, and perform either training or evaluation with its specific algorithm.
  The execution mode is specified by the <is_train> argument:
    * If <is_train> is True, then the learner will train a model with specified data & network
      architecture. The model will be saved to local files periodically.
    * If <is_train> is False, then the learner will restore a model from local files and
      measure its performance on the evaluation subset.

  All functions marked with "@abstractmethod" must be explicitly implemented in the sub-class.
  """

  def __init__(self, sm_writer, model_helper):
    """Constructor function.

    Args:
    * sm_writer: TensorFlow's summary writer
    * model_helper: model helper with definitions of model & dataset
    """

    # initialize attributes
    self.sm_writer = sm_writer
    self.data_scope = 'data'
    self.model_scope = 'model'

    # obtain the function interface provided by the model helper
    self.build_dataset_train = model_helper.build_dataset_train
    self.build_dataset_eval = model_helper.build_dataset_eval
    self.forward_train = model_helper.forward_train
    self.forward_eval = model_helper.forward_eval
    self.calc_loss = model_helper.calc_loss
    self.setup_lrn_rate = model_helper.setup_lrn_rate
    self.warm_start = model_helper.warm_start
    self.dump_n_eval = model_helper.dump_n_eval
    self.model_name = model_helper.model_name
    self.dataset_name = model_helper.dataset_name
    self.forward_w_labels = model_helper.forward_w_labels

    # checkpoint path determined by model's & dataset's names
    self.ckpt_file = 'models_%s_at_%s.tar.gz' % (self.model_name, self.dataset_name)

  @abstractmethod
  def train(self):
    """Train a model and periodically produce checkpoint files.

    Model parameters should be saved periodically for future evaluation.
    """
    pass

  @abstractmethod
  def evaluate(self):
    """Restore a model from the latest checkpoint files and then evaluate it."""
    pass

  @property
  def vars(self):
    """List of all global variables."""
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope)

  @property
  def trainable_vars(self):
    """List of all trainable variables."""
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope)

  @property
  def update_ops(self):
    """List of all update operations."""
    return tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope)
