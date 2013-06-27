import os, os.path
import sys
import numpy
from scipy import optimize
import pickle
from galpy.util import bovy_plot
from matplotlib import pyplot, cm
from selectFigs import _squeeze
import bovy_mcmc
from plotOverview import expcurve
def plotSurf(savefilename,plotfilename,kz=False):
    #Read surface densities
    #First read the surface densities
    if os.path.exists(savefilename):
        surffile= open(savefilename,'rb')
        surfrs= pickle.load(surffile)
        surfs= pickle.load(surffile)
        surferrs= pickle.load(surffile)
        kzs= pickle.load(surffile)
        kzerrs= pickle.load(surffile)
        surffile.close()
    else:
        raise IOError("savefilename with surface-densities has to exist")
    if True:#options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif False:#options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    indx= numpy.isnan(surfrs)
    indx[50]= True
    indx[57]= True
    indx= True - indx
    surfrs= surfrs[indx]
    surfs= surfs[indx]
    surferrs= surferrs[indx]
    kzs= kzs[indx]
    kzerrs= kzerrs[indx]
    if kz:
        surfs= kzs
        surferrs= kzerrs
        ylabel=r'$K_{Z}(R,|Z| = 1.1\,\mathrm{kpc})\ (2\pi G\,M_\odot\,\mathrm{pc}^{-2})$'
    else:
        ylabel=r'$\Sigma(R,|Z| \leq 1.1\,\mathrm{kpc})\ (M_\odot\,\mathrm{pc}^{-2})$'
    fehs= fehs[indx]
    afes= afes[indx]
    #Plot
    colormap = cm.jet
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(surfrs,surfs,c=afes,marker='o',scatter=True,
                        xlabel=r'$R\ (\mathrm{kpc})$',cmap='jet',
                        ylabel=ylabel,
                        xrange=[4.,10.],
                        yrange=[10,1050.],
                        semilogy=True,
                        edgecolor='none',zorder=10,s=50.)
    for ii in range(len(afes)):
        pyplot.errorbar(surfrs[ii],
                        surfs[ii],
                        yerr=surferrs[ii],
                        elinewidth=1.,capsize=3,
                        linestyle='none',zorder=5,
                        color=colormap(_squeeze(afes[ii],
                                                numpy.amax([numpy.amin(afes)]),
                                                numpy.amin([numpy.amax(afes)]))))
    #Fit exponential, plot
    trs= numpy.linspace(4.3,9.,1001)
    exp_params= optimize.fmin_powell(expcurve,
                                     numpy.log(numpy.array([72.,2.5])),
                                     args=(surfrs,surfs,surferrs))
    pyplot.plot(trs,numpy.exp(exp_params[0]-(trs-8.)/numpy.exp(exp_params[1])),
                '-',color='0.5',lw=1.,zorder=0)
    print numpy.exp(exp_params)
    step= [0.1,0.1]
    pdf_func= lambda x,y,z,u: - expcurve(x,y,z,u)
    funcargs= (surfrs,surfs,surferrs)
    isDomainFinite= [[False,False],[False,False]]
    domain= [[0.,0.],[0.,0.]]
    thesesamples= bovy_mcmc.markovpy(exp_params,
                                     step,
                                     pdf_func,
                                     funcargs,
                                     isDomainFinite=isDomainFinite,
                                     domain=domain,
                                     nsamples=10000)
    for kk in range(len(exp_params)):
        xs= numpy.array([s[kk] for s in thesesamples])
        print numpy.mean(xs), numpy.std(xs)
    bovy_plot.bovy_end_print(plotfilename)

if __name__ == '__main__':
    plotSurf(sys.argv[1],sys.argv[2],kz=len(sys.argv) == 4)
