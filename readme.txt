This code is based on both https://github.com/ej0cl6/deep-active-learning and https://github.com/JordanAsh/badge. 
The entry point is experiments.py, which allows the user to specify: 
- the number of labeled samples to start with (nStart),
- the number of samples to query at each round (nQuery), 
- the number of labeled samples to end at (nEnd),
- the experiment name to use for saving run data (savefile),
- a number <50000 to decrease the dataset to (DEBUG),
- as well as other parameters.

Command to retain output:
https://unix.stackexchange.com/a/680833

An experiment can be executed with a command like:
    python experiments.py --model resnet --data CIFAR10 --nStart 1000 --nQuery 1200 --nEnd 25000 --DEBUG 25000 --backwardSteps 0 --aug 1 --savefile <insertname>

or for small test-case purposes:
    python experiments.py --model resnet --data CIFAR10 --nStart 10 --nQuery 120 --nEnd 250 --DEBUG 250 --backwardSteps 0 --aug 1 --savefile <insertname>