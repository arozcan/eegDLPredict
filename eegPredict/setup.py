from distutils.core import setup

setup(
    name='EEGLearn',
    version='1.0',
    packages=['eegpredict'],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'theano', 'lasagne'],
    url='',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Ahmet Remzi Özcan',
    description='EEG Seizure Prediction using Deep Learning'
)
