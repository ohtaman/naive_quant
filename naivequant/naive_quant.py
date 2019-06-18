import logging
import tempfile

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


logger = logging.getLogger(__name__)


def select_quantization_variables(variables):
    return [
        v for v in variables
        if (
            v.name.endswith('act_quant/min:0')
            or v.name.endswith('act_quant/max:0')
            or v.name.endswith('weights_quant/min:0')
            or v.name.endswith('weights_quant/max:0')
        )
    ]


def quantize_in_training_phase(model_file, ckpt, inputs):
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        K.set_session(sess)
        with graph.as_default():
            K.set_learning_phase(1)
            model = keras.models.load_model(model_file, compile=False)
            original_preds = model.predict(inputs)
            
            tf.contrib.quantize.create_training_graph(input_graph=graph, quant_delay=0)
            sess.run(tf.global_variables_initializer())

            quant_vars = select_quantization_variables(tf.global_variables())
            for v in quant_vars:
                logger.debug(f'quatization variable added: {v.name}')

            model.compile(
                optimizer=keras.optimizers.SGD(0),
                loss='mean_squared_error'
            )

            logger.info('start to optimize quantization stats.')
            model.fit(inputs, original_preds, epochs=1, verbose=0)
            
            for v in quant_vars:
                logger.debug(f'quantization stats: {v.name} = {sess.run(v)}')
                    
        saver = tf.train.Saver()
        logger.info('save checkpoints to "{ckpt}"')
        saver.save(sess, ckpt)


def build_evaluation_graph(model_file, ckpt):
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:   
        K.set_session(sess)
        with graph.as_default():
            K.set_learning_phase(0)
            model = keras.models.load_model(model_file, compile=False)
            
            tf.contrib.quantize.create_eval_graph(input_graph=graph)
            graph_def = graph.as_graph_def()
            
            saver = tf.train.Saver()
            saver.restore(sess, ckpt)
            
            graph_def = tf.graph_util.remove_training_nodes(graph_def)
            graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                graph_def,
                [o_.op.name for o_ in model.outputs]
            )
    
        return graph_def, model.inputs, model.outputs


def convert_to_tflite(graph_def, inputs, outputs, input_ranges=None, default_ranges_stats=(-6, 6)):
    if input_ranges is None:
        input_ranges = [(-1, 1) for _ in inputs]
    
    converter = tf.lite.TFLiteConverter(graph_def, inputs, outputs)
    converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {
        i_: (int(-min_*255), 255./(max_ - min_))
        for i_, (min_, max_) in zip(input_arrays, input_ranges)
    }
    converter.default_ranges_stats = default_ranges_stats
    return converter.convert()


def convert(model_file, representative_dataset, input_ranges=None, default_ranges_stats=(-6, 6)):
    with tempfile.TemporaryDirectory() as ckpt:
        quantize_in_training_phase(model_file, ckpt, representative_dataset)
        graph_def, inputs, outputs = build_evaluation_graph(model_file, ckpt)
        tflite_model = convert_to_tflite(graph_def, inputs, outputs, input_ranges, default_ranges_stats)

    return tflite_model