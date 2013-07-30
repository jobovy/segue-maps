import os, os.path
import sys
import math
import numpy
import pickle
from scipy import maxentropy, integrate, special
from galpy import potential
from galpy.util import bovy_plot, save_pickles
import monoAbundanceMW
import AnDistance
from pixelFitDF import _REFV0, _REFR0, setup_potential, read_rawdata
from pixelFitDens import pixelAfeFeh
from calcDFResults import setup_options
from matplotlib import pyplot
def plotMAPMassScaleLength(plotfilename):
    #First calculate the mass-weighted scale length
    fehs= monoAbundanceMW.fehs()
    afes= monoAbundanceMW.afes()
    #Load the samples of masses and scale lengths
    #Savefile
    denssamplesfile= '../fits/pixelFitG_DblExp_BigPix0.1_1000samples.sav'
    masssamplesfile= '../fits/pixelFitG_Mass_DblExp_BigPix0.1_simpleage_1000samples.sav'
    if os.path.exists(denssamplesfile):
        savefile= open(denssamplesfile,'rb')
        denssamples= pickle.load(savefile)
        savefile.close()
        denserrors= True
    else:
        raise IOError("%s file with density samples does not exist" % denssamplesfile)
    if os.path.exists(masssamplesfile):
        savefile= open(masssamplesfile,'rb')
        masssamples= pickle.load(savefile)
        savefile.close()
    else:
        raise IOError("%s file with mass samples does not exist" % masssamplesfile)
    #Load the relative distance factors
    options= setup_options(None)
    raw= read_rawdata(options)
    binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
    #Get fehs from monoAb
    distfac= numpy.zeros_like(fehs)
    for ii in range(len(fehs)):
        #Get the relevant data
        data= binned(fehs[ii],afes[ii])
        distfac[ii]= AnDistance.AnDistance(data.dered_g-data.dered_r,
                                           data.feh)
    nsamples= 100
    rs= numpy.linspace(2.,11.5,101)
    rds= numpy.zeros((101,nsamples))
    rndindx= numpy.random.permutation(len(masssamples[108]))[0:nsamples]
    for jj in range(nsamples):
        hrs= numpy.zeros_like(fehs)
        mass= numpy.zeros_like(fehs)
        kk, ii= 0, 0
        while kk < len(denssamples):
            if denssamples[kk] is None: 
                kk+= 1
                continue
            hrs[ii]= numpy.exp(denssamples[kk][rndindx[jj]][0])*distfac[ii]*8.
            mass[ii]= masssamples[kk][rndindx[jj]] 
            ii+= 1
            kk+= 1
        #hrs[hrs > 3.5]= 3.5
        #Effective density profile
        tdens= numpy.zeros_like(rs)
        mass/= numpy.sum(mass)
        for ii in range(len(rs)):
            tdens[ii]= numpy.sum(mass*numpy.exp(-(rs[ii]-8.)/hrs))
        tdens= numpy.log(tdens)
        rds[:,jj]= -(rs[1]-rs[0])/(numpy.roll(tdens,-1)-tdens)
        #for ii in range(len(rs)):
        #    rds[ii,jj]= numpy.exp(numpy.sum(numpy.log(hrs)*mass*numpy.exp(-(rs[ii]-8.)/hrs))/numpy.sum(mass*numpy.exp(-(rs[ii]-8.)/hrs)))
    rds= rds[:-1,:]
    rs= rs[:-1]
    #Now plot
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(rs,numpy.median(rds,axis=1),'k-',
                        xrange=[0.,12.],
                        yrange=[0.,4.],
                        xlabel=r'$R\ (\mathrm{kpc})$',
                        ylabel=r'$\mathrm{effective\ disk\ scale\ length\,(kpc)}$',
                        zorder=10)
    #Find 68 range at each r
    nsigs= 1
    rcsigs= numpy.zeros((len(rs),2*nsigs))
    fs= rds
    for ii in range(nsigs):
        for jj in range(len(rs)):
            thisf= sorted(fs[jj,:])
            thiscut= 0.5*special.erfc((ii+1.)/math.sqrt(2.))
            rcsigs[jj,2*ii]= thisf[int(math.floor(thiscut*nsamples))]
            thiscut= 1.-thiscut
            rcsigs[jj,2*ii+1]= thisf[int(math.floor(thiscut*nsamples))]
    colord, cc= (1.-0.75)/(nsigs+1.), 1
    nsigma= nsigs
    pyplot.fill_between(rs,rcsigs[:,0],rcsigs[:,1],color='0.75')
    while nsigma > 1:
        pyplot.fill_between(rs,rcsigs[:,cc+1],rcsigs[:,cc-1],
                            color='%f' % (.75+colord*cc))
        pyplot.fill_between(rs,rcsigs[:,cc],rcsigs[:,cc+2],
                            color='%f' % (.75+colord*cc))
        cc+= 2
        nsigma-= 1
    #pyplot.fill_between(rs,numpy.amin(rds,axis=1),numpy.amax(rds,axis=1),
    #                    color='0.50')
    pyplot.errorbar([7.],[2.15],
                    xerr=[2.],
                    yerr=[0.14],
                    elinewidth=1.,capsize=3,
                    linestyle='none',zorder=15,
                    marker='o',color='k')
    bovy_plot.bovy_text(4.5,0.25,r'$\mathrm{Prediction\ from\ star\ counts}$'
                        +'\n'
                        +r'$\mathrm{Dynamical\ measurement}$',
                        size=14.,
                        ha='left')
    bovy_plot.bovy_plot([2.82,4.0],[0.53,0.53],'k-',overplot=True)
    pyplot.fill_between(numpy.array([2.8,4.0]),
                        numpy.array([0.45,0.45]),
                        numpy.array([0.61,0.61]),
                        color='0.75')
    pyplot.errorbar([3.5],[0.3],yerr=[0.1],xerr=[0.4],
                     color='k',marker='o',ls='none')
    bovy_plot.bovy_end_print(plotfilename)
                
if __name__ == '__main__':
    plotMAPMassScaleLength(sys.argv[1])
if __name__ == '__main__':
    plotMAPMassScaleLength(sys.argv[1])
