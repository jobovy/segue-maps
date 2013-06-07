import os, os.path
import sys
import numpy
import pickle
from scipy import maxentropy, integrate
from galpy import potential
from galpy.util import bovy_plot, save_pickles
from pixelFitDF import _REFV0, _REFR0, setup_potential
from calcDFResults import setup_options
from matplotlib import pyplot, cm
_NRDS= 31
_NFHS= 32
_ZH= 400.
_DLNVCDLNR= 0.
_VC= 230.
def plotSurfRdfh(plotfilename):
    #Calculate the surface density profile for each trial potential, then plot in 2D
    if '.png' in plotfilename:
        savefilename= plotfilename.replace('.png','.sav')
    elif '.ps' in plotfilename:
        savefilename= plotfilename.replace('.ps','.sav')
    if not os.path.exists(savefilename):
        options= setup_options(None)
        options.potential= 'dpdiskplhalofixbulgeflatwgasalt'
        options.fitdvt= False
        rs= numpy.array([5.,8.,11.])
        rds= numpy.linspace(2.,3.4,_NRDS)
        fhs= numpy.linspace(0.,1.,_NFHS)
        surfz= numpy.zeros((len(rs),len(rds),len(fhs)))+numpy.nan
        ro= 1.
        vo= _VC/ _REFV0
        dlnvcdlnr= _DLNVCDLNR
        zh= _ZH
        for jj in range(len(rds)):
            for kk in range(len(fhs)):
                #Setup potential to calculate stuff
                potparams= numpy.array([numpy.log(rds[jj]/8.),vo,numpy.log(zh/8000.),fhs[kk],dlnvcdlnr])
                try:
                    pot= setup_potential(potparams,options,0,returnrawpot=True)
                except RuntimeError:
                    continue
                for ii in range(len(rs)):
                    surfz[ii,jj,kk]= 2.*integrate.quad((lambda zz: potential.evaluateDensities(rs[ii]/8.,zz,pot)),0.,options.height/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro
        #Save
        save_pickles(savefilename,rs,rds,fhs,surfz)
    else:
        savefile= open(savefilename,'rb')
        rs= pickle.load(savefile)
        rds= pickle.load(savefile)
        fhs= pickle.load(savefile)
        surfz= pickle.load(savefile)
        savefile.close()
    #Now plot
    bovy_plot.bovy_print()
    data = numpy.ma.masked_invalid(surfz[1,:,:])
    bovy_plot.bovy_dens2d(data.filled(data.mean()).T,
                          origin='lower',
                          cmap='jet',
                          colorbar=True,
                          shrink=0.775,
                          xlabel=r'$\mathrm{disk\ scale\ length}\,(\mathrm{kpc})$',
                          ylabel=r'$\mathrm{relative\ halo\ contribution\ to}\ V^2_c(R_0)$',
                          zlabel=r'$\Sigma(R_0,|Z| \leq 1.1\,\mathrm{kpc})\, (M_\odot\,\mathrm{pc}^{-2})$',
                          xrange=[rds[0]-(rds[1]-rds[0])/2.,
                                  rds[-1]+(rds[1]-rds[0])/2.],
                          yrange=[fhs[0]-(fhs[1]-fhs[0])/2.,
                                  fhs[-1]+(fhs[1]-fhs[0])/2.])
    #Fix bad data
    bad_data = numpy.ma.masked_where(~data.mask, data.mask)
    bovy_plot.bovy_dens2d(bad_data.T,
                          origin='lower',
                          interpolation='nearest',
                          cmap=cm.gray_r,
                          overplot=True,
                          
                          xrange=[rds[0]-(rds[1]-rds[0])/2.,
                                  rds[-1]+(rds[1]-rds[0])/2.],
                          yrange=[fhs[0]-(fhs[1]-fhs[0])/2.,
                                  fhs[-1]+(fhs[1]-fhs[0])/2.])
    #Overlay contours of sigma at other R
    bovy_plot.bovy_dens2d(surfz[0,:,:].T,origin='lower',
                          xrange=[rds[0]-(rds[1]-rds[0])/2.,
                                  rds[-1]+(rds[1]-rds[0])/2.],
                          yrange=[fhs[0]-(fhs[1]-fhs[0])/2.,
                                  fhs[-1]+(fhs[1]-fhs[0])/2.],
                          overplot=True,
                          justcontours=True,
                          contours=True,
                          cntrcolors='k',
                          cntrls='-')
    bovy_plot.bovy_dens2d(surfz[2,:,:].T,origin='lower',
                          xrange=[rds[0]-(rds[1]-rds[0])/2.,
                                  rds[-1]+(rds[1]-rds[0])/2.],
                          yrange=[fhs[0]-(fhs[1]-fhs[0])/2.,
                                  fhs[-1]+(fhs[1]-fhs[0])/2.],
                          overplot=True,
                          justcontours=True,
                          contours=True,
                          cntrcolors='w',
#                          cntrlabel=True,
                          cntrls='--')
    #Add labels
    bovy_plot.bovy_text(r'$\Sigma(R=5\,\mathrm{kpc})$'
                        +'\n'
                        +r'$\Sigma(R=11\,\mathrm{kpc})$',
                        bottom_left=True,size=14.)
    bovy_plot.bovy_plot([2.575,2.8],[0.15,0.31],'-',color='0.5',overplot=True)
    bovy_plot.bovy_plot([2.625,2.95],[0.06,0.1525],'-',color='0.5',overplot=True)
    bovy_plot.bovy_end_print(plotfilename)
    return None

if __name__ == '__main__':
    plotSurfRdfh(sys.argv[1])
