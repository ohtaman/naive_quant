# Naive Quant

A naive tool for generate EdgeTPU-ready quantized tflite file.

## Limitations

- This tool does not support TensorFlow 2.x
- Not all keras models are supported since this tool uses Quantization Aware Training tool internally and the quantization process is so complicated.

## Installation

```bash
$ pip install git+https://github.com/ohtaman/naive_quant.git
```

## Usage

```python

import naivequant as nq

...

tflite_file = n.convert(
    keras_model_file,
    representative_dataset, 
    input_ranges,
    default_ranges_stats
)

with open('model.tflite', 'wb') as o_:
    o_.write(tflite_file)

```