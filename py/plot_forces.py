import os
import sys
import numpy
from galpy.util import bovy_plot
from galpy import potential
from pixelFitDF import setup_potential, _REFV0, _REFR0
from calcDFResults import setup_options
def plot_forces():
    potparams= numpy.array([numpy.log(2.15/_REFR0),
                            1.,
                            numpy.log(0.3/_REFR0),
                            0.307169914244,
                            -1.79107550983])
    options= setup_options(None)
    options.potential= 'dpdiskplhalofixcutbulgeflatwgasalt'
    pot= setup_potential(potparams,options,0,returnrawpot=True)
    bovy_plot.bovy_print(fig_width=8.,fig_height=5.)
    potential.plotDensities(pot,rmin=-15./8.,rmax=15./8.,nrs=100,
                            zmin=-10./8,zmax=10./8.,nzs=101,ncontours=10,
                            aspect=5./5.,log=True)
    
    bovy_plot.bovy_end_print('/home/bovy/Desktop/test.png')
    pass

if __name__ == '__main__':
    plot_forces()
