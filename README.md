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
It allows to label an unsegmented sequence by adding a "blank" label which is ignored at final prediction.

<https://www.cs.toronto.edu/~graves/icml_2006.pdf>

### BLSTM
BLSTM (Bidirectional Long-Short Term Memory) cells are units of RNN layers able to remember features over time, and forget elements which are not useful for the sequence . A neural layer of LSTM can give sequences by using a label which means the end of the sequence.

### CNN
Before using BLSTM layers, we process the images to obtain features of them thanks to a convolutional network.

### Architecture chosen
The architecture chosen for this project consists in 6 Conv layers followed by an AveragePooling and 3 BLSTM, then a dense layer is added with a softmax activation function to classify each element of the sequence. This architecture is trained thanks to CTC.
Each part of the label (note name, octave and rythm) are classified by a head composed of the 2 lasts BLSTM layers and a Softmax. The deep layers of the model are common for the three classifiers.


## Results
At the moment, Notes names and octaves network are trained (not rythms).
On the evaluation dataset, here are the results obtained. error rate is the mean of the label error rate for each image.

|label error rate  | note name | octave | rythms|
| ------ | ------ | ------ | ------ |
|6 CNN + 3 BLSTM (on augmented datas) | 0.069 | 0.1 | 0.025|
|3 BLSTM  | 0.1271 | 0.1516 | //|


Here are some examples of images with predictions associated for the model with only the three BSLTM layers:

![image 1](https://github.com/GaetanBaert/OMR_deep/blob/master/images/100508_0.png)


prediction (notename_octave):

 ` B-_5 rest_rest D_6 D_6 C_| D_6 rest_5 rest_rest F_rest B-_4 D_5 A-_5 B-_5 |_5 F_| A-_5 E-_5 F_5 rest_5 E-_rest D_5 E-_5 E-_5 D_5 A-_5 B-_4 D_4 B-_5 4 `


reality :

` B-_4 rest_rest D_5 D_5 C#_5 D_5 rest_rest rest_rest F_4 B-_4 D-_5 A-_4 B-_4 |_| F_4 A-_4 E-_4 F_4 rest_rest E-_4 D-_4 F-_4 E-_4 D-_4 A-_3 A-_3 D-_4 B-_3 `

We can see there is some troubles with exotic keys (G key with an octave offset, the little 8 on the image), but the notes names are good. There is some deletion/addition errors on octave labels(a | at the start). It's interesting to see that notes names works for flats in key signatures

![image 1](https://github.com/GaetanBaert/OMR_deep/blob/master/images/101334_1.png)


prediction :

`A_4 G#_4 B_4 |_| C#_4 F_4 |_| F#_4 C#_4 |_| B_4 G#_4 A_4 |_| E_4 F#_4 D_4 D#_4 |_| C_4 B-_4 G_4 |_| G_4 |_| E_4 |_| C#_5 |_| C#_5 |_| C#_5 |_| A_4 `


reality :

`A_4 G#_4 B_4 |_| C#_4 F_4 |_| F_4 C#_4 |_| B_4 G#_4 A_4 |_| E_4 F#_4 D_4 D#_4 |_| C_4 B-_4 G_4 |_| G_4 |_| E_4 |_| C#_5 |_| C-_5 |_| C#_5 |_| A_4 `

Here the result is better (perfect match for octaves) and the errors on notes names are only between sharps and flats.






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
