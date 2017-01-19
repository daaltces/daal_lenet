#==============================================================
#
# SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
# http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
#
# Copyright 2016 Intel Corporation
#
# THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
#
# =============================================================


DAAL_PATH = "$(DAALROOT)/lib/intel64_lin"
DAAL_LIBS := -ldaal_core -ldaal_thread

TBB_PATH = "$(DAALROOT)/../tbb/lib/intel64_lin/gcc4.7"
EXT_LIBS := -ltbb -ltbbmalloc -lpthread -ldl 

COPTS := -std=c++11 -m64 -Wall -w
LOPTS := -L$(DAAL_PATH) $(DAAL_LIBS) -L$(TBB_PATH) $(EXT_LIBS)

CC = g++

daal_lenet.exe: ./daal_lenet.cpp
	$(CC) $(COPTS) $< -o $@ $(LOPTS)

clean:
	rm ./daal_lenet.exe
