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
