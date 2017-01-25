#Mezzomorphic Recurrent Neural Network
> Deep Learning Music Generator. 

Mezzomorphic is a web application that uses Rails and Python to generate music via a recurrent neural network.

This network is used in conjuction with a Rails Api [controller](https://github.com/louiehchen/mezzo_backend) and rails 5 Application [front-end](https://github.com/louiehchen/mezzo_frontend).


The Mezzomorphic network trains on .wav files. It uses pattern recognition to output a set of bytes that create a brand new computer generated sound. 


##Installation
```
pip3 install keras
pip3 install sklearn
```

*If you don't already have Python3 and Pip installed you can find installation instructions [here](https://www.python.org/downloads/).*


##Usage
```
python3 main.py [input file(s)] [output file]
```

Run main.py using python3 with the arguments of the file you wish to train the network on and the name of the file you'd like the network to output.
* *Input files must be in WAVE_FORMAT_PCM:0x0001*
* *Input file can also be a folder. The netwrok will use all .wav files in the folder.*
* *Output file name is optional. If excluded will be named output_file.wav by default*


##Time Considerations
Running this neural network on wav files in any significant length takes considerable processing(CPU/GPU) power and time. For example, a 15 second clip took about two days to train, convert, and generate a new clip on a typical laptop. 


##Other Resources
To see some other neural networks in action and for more information check this youtube [playlist](https://www.youtube.com/watch?v=b99UVkWzYTQ&list=PLjJh1vlSEYgvGod9wWiydumYl8hOXixNu) and [article](http://machinelearningmastery.com/best-machine-learning-resources-for-getting-started/) out. 
For more information on recurrent neural networks, click [here](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
For more information on keras, check the [docs](https://keras.io/).



#Team
* [Aaron Jacobson](https://github.com/SudoSanauu)
* [Alex Mcleod](https://github.com/mcleodaj)
* [Alexandria Nelson](https://github.com/Alex-CAN)
* [Louie Chen](https://github.com/louiehchen)
* [Vivi Nguyen](https://github.com/CatonNip)
