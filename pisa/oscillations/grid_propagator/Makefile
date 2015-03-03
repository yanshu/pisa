# Makefile
#
# IMPORTANT! If you don't rename EarthDensity.py -> <something
# else>.py, then bad things/name collisions happen!
#

# Needed to create the __init__.py
#SHELL:=/bin/bash

CXXFLAGS += -I. -Wall -Wno-c++11-compat-deprecated-writable-strings
CXX = g++ -O3

#############################
# NEEDED FOR PYTHON DLL
PYLIB:=$(shell python -c 'from distutils import sysconfig; print sysconfig.get_config_var("LIBDIR")')
PYINC:=$(shell python -c 'from distutils import sysconfig; print sysconfig.get_config_var("INCLUDEPY")')
#PYLIB = /Users/timothyarlen/anaconda/pkgs/python-2.7.8-1/lib/python2.7/
#PYINC = /Users/timothyarlen/anaconda/pkgs/python-2.7.8-1/include/python2.7/
PYFLAGS = -L$(PYLIB) -lpython2.7

NUMPYINC:=$(shell python -c 'import numpy; print numpy.get_include()')
#NUMPYINC = /usr/local/lib/python2.7/site-packages/numpy/core/include/

# Depending on your OS/compiler, you may need a different for a dynamic lib for c++
DLLFLAG = -dynamiclib
SWIG = $(shell which swig)
#############################

%.o : %.cpp
	$(RM) $@
	$(CXX) -c $(CXXFLAGS) -o $@ $*.cpp

OBJS = EarthDensity.o GridPropagator.o

targets = _GridPropagator.so GridProp.py

# Now generate the python dll
_GridPropagator.so: GridPropagator_wrap.o GridPropagator.o
	$(CXX) $(DLLFLAG) GridPropagator_wrap.o $(OBJS) $(PYFLAGS) -o $@
	touch __init__.py

GridPropagator_wrap.o: GridPropagator_wrap.cxx $(OBJS)
	$(CXX) $(CXXFLAGS) GridPropagator_wrap.cxx -c -g -I$(PYINC) -I$(NUMPYINC)

GridPropagator_wrap.cxx: GridPropagator.i
	$(SWIG) -v -c++ -python GridPropagator.i

GridProp.py: GridPropagator.i
	$(SWIG) -v -c++ -python GridPropagator.i

.PHONY: all
all: $(targets)

.PHONY: clean
clean:
	$(RM) $(targets) *.o *.a *.so *.pyc *~ *.cxx

emptyrule: $(targets)
