## About this project

This program is the product of a school project on general-purpose GPU computing.
It can build and query simple rainbow table and is accelerated with OpenCL.

## How to build and run

CMake is used:

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    $ ./rt-build -h

Or you can just use the `./run` script, which automatically generates the
`build` directory and runs the programs:

    $ ./run rt-build -h
    $ ./run rt-lookup -h
    $ ./run rt-benchmarks -h
