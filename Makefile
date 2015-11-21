# You may need to edit this file to reflect the type and capabilities of your system.
# The defaults are for a Linux system and may need to be changed for other systems (eg. Mac OS X).


CXX=g++

INPUT = main_ok.cpp


INCLUDE_FLAG = -c -I "../"  -O3 -pthread -std=c++11 -Wl,--no-as-needed



CXXFLAGS = $(INCLUDE_FLAG)

main.o: main_ok.cpp
	$(CXX) $(CXXFLAGS) $(INPUT)  -o $@ 


.PHONY: clean

clean:
	rm -f out main.o

