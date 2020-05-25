"""Utility functions for learning rates."""

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def setup_lrn_rate_piecewise_constant(global_step, batch_size, idxs_epoch, decay_rates):
  """Setup the learning rate with piecewise constant strategy.

  Args:
  * global_step: training iteration counter
  * batch_size: number of samples in each mini-batch
  * idxs_epoch: indices of epoches to decay the learning rate
  * decay_rates: list of decaying rates

  Returns:
  * lrn_rate: learning rate
  """

  # adjust interval endpoints w.r.t. FLAGS.nb_epochs_rat
  idxs_epoch = [idx_epoch * FLAGS.nb_epochs_rat for idx_epoch in idxs_epoch]

  # setup learning rate with the piecewise constant strategy
  lrn_rate_init = FLAGS.lrn_rate_init * batch_size / FLAGS.batch_size_norm
  nb_batches_per_epoch = float(FLAGS.nb_smpls_train) / batch_size
  bnds = [int(nb_batches_per_epoch * idx_epoch) for idx_epoch in idxs_epoch]
  vals = [lrn_rate_init * decay_rate for decay_rate in decay_rates]
  lrn_rate = tf.train.piecewise_constant(global_step, bnds, vals)

  return lrn_rate

def setup_lrn_rate_exponential_decay(global_step, batch_size, epoch_step, decay_rate):
  """Setup the learning rate with exponential decaying strategy.

  Args:
  * global_step: training iteration counter
  * batch_size: number of samples in each mini-batch
  * epoch_step: epoch step-size for applying the decaying step
  * decay_rate: decaying rate

  Returns:
  * lrn_rate: learning rate
  """

  # adjust the step size & decaying rate w.r.t. FLAGS.nb_epochs_rat
  epoch_step *= FLAGS.nb_epochs_rat

  # setup learning rate with the exponential decay strategy
  lrn_rate_init = FLAGS.lrn_rate_init * batch_size / FLAGS.batch_size_norm
  batch_step = int(FLAGS.nb_smpls_train * epoch_step / batch_size)
  lrn_rate = tf.train.exponential_decay(
    lrn_rate_init, tf.cast(global_step, tf.int32), batch_step, decay_rate, staircase=True)

  return lrn_rate
