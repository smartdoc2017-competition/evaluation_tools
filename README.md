# ICDAR 2017 Competition SmartDoc-reconstruction
## Evaluation tools
This repository contains the evaluation tools used to rank the participant results during the competition.

### Language and dependencies
This example method is wrote using Python 2.7+ for maximal compatibility with existing image processing libraries.

It depends on 3 libraries:
* **Numpy** (any recent version should work): the famous package for scientific computing -- http://www.numpy.org/
* **OpenCV** (version 2.10+ but not 3.x): the famous computer vision library -- http://opencv.org/
* **PuLP**  (version 1.6.0+): Python linear programming library -- https://pythonhosted.org/PuLP/

As this methods makes use of SIFT for image matching, we believe the 2.X series are easier to use.
If you need OpenCV 3.X support, it may be easy to fix the incompatible few lines.

If you use a virtual environment to develop in Python, we recommend to use VirtualEnv Wrapper (http://virtualenvwrapper.readthedocs.io/) and to give the virtual environment access to the global site-packages using a command like:
~~~
$ mkvirtualenv --system-site-packages smartdoc17
~~~

This example method was successfully run within a Docker container. Any decent container with OpenCV 2.10+ and Python should work.

Getting PuLP to work should be a matter of
~~~
$ pip install pulp
$ pulptest
~~~

Installing a separate linear programming solver might be helpful if you encounter issues using the default solver packaged pulp. We successfully used GLPK (GNU Linear Programming Kit).
On Debian-based / Ubuntu, the development files required can be installed with:
~~~
$ sudo apt install libglpk-dev
~~~


### Installation
Simply copy the files of this repository.

### Update
Reinstall from https://github.com/smartdoc2017-competition/evaluation_tools

### Usage
Run the following command to print out the help:
~~~
$ python main.py -h
~~~

To run the method, you can use:
~~~
$ python main.py --debug --gui \
	--output path/to/result.json \
	path/to/sampleNN/ground-truth.png \
	path/to/sampleNN/reference_frame_??_dewarped.png

~~~

`--debug` and `--gui` activate debug output and graphical interface, respectively. The GUI slows the method down.

**The global evaluation of the quality of the restoration is indicated in the `mssim_best` of the `result.json` file.**

The global evaluation will average all those measures for a given participant.

### Questions
Ask us any question about the competition at: icdar (dot) smartdoc (at) gmail (dot) com


### License
The LICENCE file contains the details about the MIT license used.

In a nutshell, it says that you can do whatever you want with the code of this repository as long as:
* you don't hold us liable for anything
* you credit our work properly


