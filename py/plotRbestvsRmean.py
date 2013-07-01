import os, os.path
import sys
import numpy
from scipy import optimize
import pickle
from galpy.util import bovy_plot
from matplotlib import pyplot, cm
from selectFigs import _squeeze
from plotOverview import expcurve
from pixelFitDens import pixelAfeFeh
def plotRbestvsRmean(savefilename,plotfilename):
    #Read surface densities
    #First read the surface densities
    if os.path.exists(savefilename):
        surffile= open(savefilename,'rb')
        surfrs= pickle.load(surffile)
        surfs= pickle.load(surffile)
        surferrs= pickle.load(surffile)
        surffile.close()
    else:
        raise IOError("savefilename with surface-densities has to exist")
    if True:#options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif False:#options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    #Load g orbits
    orbitsfile= 'gOrbitsNew.sav'
    savefile= open(orbitsfile,'rb')
    orbits= pickle.load(savefile)
    savefile.close()
    #Cut to S/N, logg, and EBV
    indx= (orbits.sna > 15.)*(orbits.logga > 4.2)*(orbits.ebv < 0.3)
    orbits= orbits[indx]
    #Load the orbits into the pixel structure
    pix= pixelAfeFeh(orbits,dfeh=0.1,dafe=0.05)
    #Now calculate meanr
    rmean= numpy.zeros_like(surfrs)
    for ii in range(len(surfrs)):
        data= pix(fehs[ii],afes[ii])
        vals= data.densrmean*8.
        if False:#True:
            rmean[ii]= numpy.mean(vals)
        else:
            rmean[ii]= numpy.median(vals)
    #Plot
    indx= numpy.isnan(surfrs)
    indx[50]= True
    indx[57]= True
    indx= True - indx
    surfrs= surfrs[indx]
    rmean= rmean[indx]
    fehs= fehs[indx]
    afes= afes[indx]
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(rmean,surfrs,c=afes,marker='o',scatter=True,
                        xlabel=r'$\mathrm{mean\ orbital\ radius\ of\ MAP}\,(\mathrm{kpc})$',
                        ylabel=r'$\mathrm{radius\ at\ which\ MAP\ measures}\ \Sigma_{1.1}\,(\mathrm{kpc})$',
                        xrange=[4.,10.],
                        yrange=[4.,10.],
                        edgecolor='none',zorder=10,s=50.)
    bovy_plot.bovy_plot([4.5,9.],[4.5,9.],'-',color='0.50',overplot=True,lw=1.)
    bovy_plot.bovy_end_print(plotfilename)

if __name__ == '__main__':
    plotRbestvsRmean(sys.argv[1],sys.argv[2])
