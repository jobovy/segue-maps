import os, os.path
import math
import numpy
import cPickle as pickle
from matplotlib import pyplot
from optparse import OptionParser
from scipy import optimize, special
from galpy.util import bovy_coords, bovy_plot
def plotSigzFunc(parser):
    (options,args)= parser.parse_args()
    if len(args) == 0:
        parser.print_help()
        return
    if os.path.exists(args[0]):#Load savefile
        savefile= open(args[0],'rb')
        params1= pickle.load(savefile)
        samples1= pickle.load(savefile)
        savefile.close()
    else:
        print "Need to give filename ..."
    if os.path.exists(args[1]):#Load savefile
        savefile= open(args[1],'rb')
        params1= pickle.load(savefile)
        samples2= pickle.load(savefile)
        savefile.close()
    else:
        print "Need to give filename ..."
        #First one
    zs= numpy.linspace(0.3,1.2,1001)
    xrange= [0.,1.3]
    yrange= [0.,60.]
    #Now plot the mean and std-dev from the posterior
    zmean= numpy.zeros(len(zs))
    nsigs= 3
    zsigs= numpy.zeros((len(zs),2*nsigs))
    fs= numpy.zeros((len(zs),len(samples1)))
    ds= zs-0.5
    for ii in range(len(samples1)):
        thisparams= samples1[ii]
        fs[:,ii]= math.exp(thisparams[1])+thisparams[2]*ds+thisparams[3]*ds**2.
    #Record mean and std-devs
    zmean[:]= numpy.mean(fs,axis=1)
    bovy_plot.bovy_print()
    xlabel=r'$|z|\ [\mathrm{kpc}]$'
    ylabel=r'$\sigma_z\ [\mathrm{km\ s}^{-1}]$'
    bovy_plot.bovy_plot(zs,zmean,'k-',xrange=xrange,yrange=yrange,
                        xlabel=xlabel,
                        ylabel=ylabel)
    for ii in range(nsigs):
        for jj in range(len(zs)):
            thisf= sorted(fs[jj,:])
            thiscut= 0.5*special.erfc((ii+1.)/math.sqrt(2.))
            zsigs[jj,2*ii]= thisf[int(math.floor(thiscut*len(samples1)))]
            thiscut= 1.-thiscut
            zsigs[jj,2*ii+1]= thisf[int(math.floor(thiscut*len(samples1)))]
    colord, cc= (1.-0.75)/nsigs, 1
    nsigma= nsigs
    pyplot.fill_between(zs,zsigs[:,0],zsigs[:,1],color='0.75')
    while nsigma > 1:
        pyplot.fill_between(zs,zsigs[:,cc+1],zsigs[:,cc-1],
                            color='%f' % (.75+colord*cc))
        pyplot.fill_between(zs,zsigs[:,cc],zsigs[:,cc+2],
                            color='%f' % (.75+colord*cc))
        cc+= 1.
        nsigma-= 1
    bovy_plot.bovy_plot(zs,zmean,'k-',overplot=True)
    #Second one
    zmean= numpy.zeros(len(zs))
    zsigs= numpy.zeros((len(zs),2*nsigs))
    fs= numpy.zeros((len(zs),len(samples2)))
    for ii in range(len(samples2)):
        thisparams= samples2[ii]
        fs[:,ii]= math.exp(thisparams[1])+thisparams[2]*ds+thisparams[3]*ds**2.
    #Record mean and std-devs
    zmean[:]= numpy.mean(fs,axis=1)
    for ii in range(nsigs):
        for jj in range(len(zs)):
            thisf= sorted(fs[jj,:])
            thiscut= 0.5*special.erfc((ii+1.)/math.sqrt(2.))
            zsigs[jj,2*ii]= thisf[int(math.ceil(thiscut*len(samples2)))]
            thiscut= 1.-thiscut
            zsigs[jj,2*ii+1]= thisf[int(math.ceil(thiscut*len(samples2)))]
    colord, cc= (1.-0.75)/nsigs, 1
    nsigma= nsigs
    pyplot.fill_between(zs,zsigs[:,0],zsigs[:,1],color='0.75')
    while nsigma > 1:
        pyplot.fill_between(zs,zsigs[:,cc+1],zsigs[:,cc-1],
                            color='%f' % (.75+colord*cc))
        pyplot.fill_between(zs,zsigs[:,cc],zsigs[:,cc+2],
                            color='%f' % (.75+colord*cc))
        cc+= 1.
        nsigma-= 1
    bovy_plot.bovy_plot(zs,zmean,'k-',overplot=True)
    bovy_plot.bovy_text(r'$-0.4 < [\mathrm{Fe/H}] < 0.5\,, \ \ -0.25 < [\alpha/\mathrm{Fe}] < 0.2$',bottom_right=True)
    bovy_plot.bovy_text(r'$-1.5 < [\mathrm{Fe/H}] < -0.5\,, \ \ 0.25 < [\alpha/\mathrm{Fe}] < 0.5$',top_left=True)
    bovy_plot.bovy_end_print(options.plotfile)
    return None

def get_options():
    usage = "usage: %prog [options] <savefilename>\n\nsavefilename= name of the file that the fit/samples will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("-o",dest='plotfile',
                      help="Name of file for plot")
    return parser

if __name__ == '__main__':
    plotSigzFunc(get_options())
