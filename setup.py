from setuptools import setup, find_packages


setup(
    name='naive_quant', 
    version='0.1.0',
    description='A naive tool for generate EdgeTPU-ready quantized tflite file',
    long_description=None,
    url='https://github.com/ohtaman/naive_quant',  # Optional
    author='ohtaman',
    author_email='ohtamans@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache License 2.0',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='tensorflow edgetpu',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
    extras_require={
        'cpu': 'tensorflow<2.0',
        'gpu': 'tensorflow-gpu<2.0'
    },
)