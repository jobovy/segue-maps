import os, os.path
import sys
import numpy
import pickle
from scipy import maxentropy, integrate
from galpy import potential
from galpy.util import bovy_plot, save_pickles
from pixelFitDF import _REFV0, _REFR0, setup_potential
from calcDFResults import setup_options
from matplotlib import pyplot
def plotPriorSurf(plotfilename):
    #Calculate the surface density profile for each trial potential, then plot the range
    if '.png' in plotfilename:
        savefilename= plotfilename.replace('.png','.sav')
    elif '.ps' in plotfilename:
        savefilename= plotfilename.replace('.ps','.sav')
    if not os.path.exists(savefilename):
        options= setup_options(None)
        options.potential= 'dpdiskplhalofixbulgeflatwgasalt'
        options.fitdvt= False
        rs= numpy.linspace(4.2,9.8,101)
        rds= numpy.linspace(2.,3.4,8)
        fhs= numpy.linspace(0.,1.,16)
        surfz= numpy.zeros((len(rs),len(rds)*len(fhs)))+numpy.nan
        ro= 1.
        vo= 1.#230./220.
        dlnvcdlnr= 0.
        zh= 400.
        for jj in range(len(rds)):
            for kk in range(len(fhs)):
                #Setup potential to calculate stuff
                potparams= numpy.array([numpy.log(rds[jj]/8.),vo,numpy.log(zh/8000.),fhs[kk],dlnvcdlnr])
                try:
                    pot= setup_potential(potparams,options,0,returnrawpot=True)
                except RuntimeError:
                    continue
                for ii in range(len(rs)):
                    surfz[ii,jj*len(fhs)+kk]= 2.*integrate.quad((lambda zz: potential.evaluateDensities(rs[ii]/8.,zz,pot)),0.,options.height/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro
        #Find minimum and maximum curves
        minsurfz= numpy.nanmin(surfz,axis=1)
        maxsurfz= numpy.nanmax(surfz,axis=1)
        #Save
        save_pickles(savefilename,rs,minsurfz,maxsurfz)
    else:
        savefile= open(savefilename,'rb')
        rs= pickle.load(savefile)
        minsurfz= pickle.load(savefile)
        maxsurfz= pickle.load(savefile)
        savefile.close()
    #Plot
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot([numpy.nan],[numpy.nan],'ko',
                        xlabel=r'$R\ (\mathrm{kpc})$',
                        ylabel=r'$\Sigma(R,|Z| \leq 1.1\,\mathrm{kpc})\ (M_\odot\,\mathrm{pc}^{-2})$',
                        xrange=[4.,10.],
                        yrange=[10,1050.],
                        semilogy=True)
    pyplot.fill_between(rs,minsurfz,maxsurfz,
                        color='0.50')
    bovy_plot.bovy_end_print(plotfilename)
    return None

if __name__ == '__main__':
    plotPriorSurf(sys.argv[1])
