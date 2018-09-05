# MCL_lite
MCL_lite is a memory-efficient implementation of Markov Clustering Algorithm for large-scale networks. In our tests, MCL_lite can cluster a network with 2 billion edges on a desktop with 16GB RAM while the original [MCL](https://micans.org/mcl/ "https://micans.org/mcl/") needs more than 100GB RAM.

## Requirement

Make sure that you have the following installed

1. Python2.7 (Recommend [Anaconda](https://www.continuum.io/downloads#linux "https://www.continuum.io/downloads#linux" ))
    1. [numpy](http://www.numpy.org/ "http://www.numpy.org/")
    2. [scipy](https://www.scipy.org/ "https://www.scipy.org/")
    3. [sklearn](http://scikit-learn.org/stable/ "http://scikit-learn.org/stable/")
    4. [numba](https://numba.pydata.org/ "https://numba.pydata.org/")

    5. Install dependency packages via pip:

        $ pip install scipy numpy scikit-learn

    6. Install dependency packages via conda:

        $ conda install scipy numpy scikit-learn


## Download

    $ git clone https://github.com/Rinoahu/MCL_lite

## Usage

    $python MCL_lite/mcl_sparse.py -i foo.xyz -I 1.5 -o foo.mcl -a 8 -m 8 -d t

-i: input network in 3 column tab-delimited format. The fist and second columns stand for nodes and third column stands for weight between these two nodes. For example: 

    N0	N1	1.5
    N0	N2	1.2
	N0	N3	1.1
	...

-I: float. inflation parameter for mcl

-a:   int. cpu number

-o:   str. name of output file

-d:   t|f. is the graph directed? Default is t(true)

-g:   int. how many gpus to use for speedup. Default is 0

-r:   t|f. resume the work after crash. Default is f(false)

-m:   int. memory usage limitation. Default is 4GB

