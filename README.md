# OMR_deep
an Optical Music Recognition (OMR) system with deep learning.

## Introduction
The objective is to build a system able to recognise notes on images.

## Dataset
The dataset is built thanks to Musescore database, only on monophonic scores (polyphonic instruments like piano are not in the dataset). The dataset is cut in three parts : train, evaluation and validation.
Scores are downloaded in musicxml format, then cut in th objective to build images with only one stave on each one.
Then, labels are extracted from the musicxml files : notes are labeled A, B, C, D, E, F, G, rest. sharps and flats are symbolized by + or - after the letter. Octaves are labeled and rythms are labeled with floats (1 is quarter note, 2 is half note, 0.5 is eigth note). bar lines are also labeled.Images are resized to all have the same height.

## Choice of the algorithm
Staves don't always have the same length and have a different number of notes. The CTC model seems to be a good option and proves it efficiency on Optical Character Recognition. OMR is a branch of OCR, with particularities : there is no words and each symbol contains two informations : rythm and tone. Here, it was decided to separe note name with octave in two separate ways. Before CTC model, we use BLSTM layers : actually, some informations depends of elements before the note, for example the key and the key signature act directly on tone.

### CTC model
The CTC model (Connectionist Temporal Classification) is an algorithm that allows to train convolutional neural network layers like RNN or, more especially LSTM layers. The main advantage of CTC is to manage the different spaces between the element of a sequence.

<https://www.cs.toronto.edu/~graves/icml_2006.pdf>

### BLSTM
BLSTM (Bidirectional Long-Short Term Memory) cells are units of a RNN layer able to remember values at different moments over time. A neural layer of LSTM can give sequences by using a label which means the end of the sequence.

### Architecture chosen
The architecture chosen for this project consists in 3 BLSTM layers of 256 neurons, then a dense layer is added with a softmax activation function to classify each element of the sequence. This architecture is trained thanks to CTC.
Each part of the label (note name, octave and rythm) are classified by a different neural network .

## Results
At the moment, Notes names and octaves network are trained (not rythms).
On the evaluation dataset, here are the results obtained. error rate is the mean of the label error rate for each image.

| | note name | octave |
| ------ | ------ | ------ |
| error rate | | 0.1516 |

Here is some examples of images with predictions associated : 

## Libraries used :
+ Numpy
+ OpenCv
+ music21 : <http://web.mit.edu/music21/>
+ Keras with Tensorflow backend

## Aknowledgments
I have to thank Robin Condat, for his help about the construction of the dataset.
I want to thank Yann Soulard and Cyprien Ruffino for their implementation of the CTC model, available here : <https://github.com/ysoullard/CTCModel>.
I want to thank the Musescore team for the dataset.
I also want to thank Eelco Van der Weel and Karen Ullrich for their paper that inspired me for thisproject : <https://arxiv.org/pdf/1707.04877.pdf>. They also proposed a script to download the Musescore dataset : <https://github.com/eelcovdw/mono-musicxml-dataset>
Finally, I want to thank Clément Chatelain for his help and INSA Rouen-Normandie for giving me the time to work on this project.
