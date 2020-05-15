import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.python.ops import io_ops
from tensorflow.python.training.saver import BaseSaverBuilder


def checkpoint_dtype_cast(in_checkpoint_file, out_checkpoint_file):
    var_list = checkpoint_utils.list_variables(tf.flags.FLAGS.init_checkpoint)

    def init_graph():
        for name, shape in var_list:
            var = checkpoint_utils.load_variable(tf.flags.FLAGS.init_checkpoint, name)
            recon_dtype = tf.float16 if var.dtype == np.float32 else var.dtype
            tf.get_variable(name, shape=shape, dtype=recon_dtype)

    init_graph()
    saver = tf.train.Saver(builder=CastFromFloat32SaverBuilder())
    with tf.Session() as sess:
        saver.restore(sess, in_checkpoint_file)
        saver.save(sess, 'tmp.ckpt')

    tf.reset_default_graph()

    init_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'tmp.ckpt')
        saver.save(sess, out_checkpoint_file)


class CastFromFloat32SaverBuilder(BaseSaverBuilder):
    # Based on tensorflow.python.training.saver.BulkSaverBuilder.bulk_restore
    def bulk_restore(self, filename_tensor, saveables, preferred_shard,
                     restore_sequentially):
        restore_specs = []
        for saveable in saveables:
            for spec in saveable.specs:
                restore_specs.append((spec.name, spec.slice_spec, spec.dtype))
        names, slices, dtypes = zip(*restore_specs)
        restore_dtypes = [tf.float32 if dtype.base_dtype==tf.float16 else dtype for dtype in dtypes]
        # print info
        for i in range(len(restore_specs)):
            print(names[i], 'from', restore_dtypes[i], 'to', dtypes[i].base_dtype)
        with tf.device("cpu:0"):
            restored = io_ops.restore_v2(
                filename_tensor, names, slices, restore_dtypes)
            return [tf.cast(r, dt.base_dtype) for r, dt in zip(restored, dtypes)]


if __name__ == '__main__':
    tf.flags.DEFINE_string("fp16_checkpoint", "sst2_output/fp16_model.ckpt", "fp16 checkpoint file")
    tf.flags.DEFINE_string("init_checkpoint", "bert_base/bert_model.ckpt", "initial checkpoint file")
    checkpoint_dtype_cast(tf.flags.FLAGS.init_checkpoint, tf.flags.FLAGS.fp16_checkpoint)