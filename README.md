# unit_square_cell

## Student Arbeit

Keywords:
 1. Multiscale Modelling
 2. Computational Homogenization
 3. Multi Field Modelling

## Source Code

*cell_geom.py*, *cell_material.py*, *cell_computation.py*

## Two Reports

There are two reports for this Forschungsmodul. Please see the
[report directory](./report/). The theory is introduced in the part 1, 
while the concepts of implementation and numerical results are discussed 
in part 2.

## Sphinx Documentation

A detailed documentation is given in the [sphx_v1 directory](sphx_v1/build/html/).
You can visit the *index.html* in the build file.

## Examples

If you need a quick overview of all the usages and examples. [example directory](example/) 
is what you are looking for. Both pdf manual and interactive ipynb files 
are included.

## Try-outs in IPython

There are some IPython notebooks testing the features and consistency in the [ipy_notebook directory](ipy_notebook/)

## Unittest

If new features are added, you can use files in [test](test/) to carry out unit 
tests.

__Please note that all the results are tested under python 3.6.7 and FEniCS 2019.1.0__

this project is revived, and adapted to the mentioned versions above

all the test is done via docker images - quay.io/fenicsproject/stable:current

one could try and test ipy-nb with the following command
`docker run --name old_unit_cell_jupy -w /home/fenics -v $(pwd):/home/fenics/shared -d -p 127.0.0.1:8888:8888 quay.io/fenicsproject/stable:current 'jupyter-notebook --ip=0.0.0.0'`