import os, os.path
import numpy
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_plot, bovy_coords
from matplotlib import pyplot
import segueSelect
from fitSigz import readData, _ARICHFEHRANGE, _APOORFEHRANGE, \
    _ARICHAFERANGE, _APOORAFERANGE, _ZSUN
def plotVelocityData(options,args):
    if options.png: ext= 'png'
    else: ext= 'ps'
    if options.metal.lower() == 'rich':
        feh= -0.15
        fehrange= _APOORFEHRANGE
    elif options.metal.lower() == 'poor':
        feh= -0.65
        fehrange= _ARICHFEHRANGE
    #Load data
    XYZ,vxvyvz,cov_vxvyvz,data= readData(metal=options.metal,
                                         sample=options.sample)
    R= ((8.-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)
    XYZ[:,2]+= _ZSUN
    if options.type.lower() == 'datavzz':
        plotx= numpy.fabs(XYZ[:,2])*1000.
        xrange=[0.,2700.]
        xlabel=r'$z\ [\mathrm{pc}]$'
    else:
        plotx= R
        xrange=[5.,14.]
        xlabel=r'$R\ [\mathrm{kpc}]$'
    #Now plot
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(plotx,vxvyvz[:,2],'k,',
                        xrange=xrange,
                        yrange=[-150.,150.],
                        xlabel=xlabel,
                        ylabel=r'$v_z\ [\mathrm{km\ s}^{-1}]$')
    #Bin and calculate mean and 1s
    nbins= 21
    if options.type.lower() == 'datavzz':
        rs= numpy.linspace(0.,2700.,nbins+1)
    else:
        rs= numpy.linspace(5.,14.,nbins+1)
    medians, smin,splus= numpy.zeros(nbins), numpy.zeros(nbins), \
        numpy.zeros(nbins)
    for ii in range(nbins):
        indx= (plotx > rs[ii])*(plotx <= rs[ii+1])
        if numpy.sum(indx) < 10.:
            medians[ii]= numpy.nan
            smin[ii]= numpy.nan
            splus[ii]= numpy.nan
            continue
        thisvz= sorted(vxvyvz[indx,2])
        #Calculate quantiles
        indx1= int(numpy.floor(0.5*len(thisvz)))
        indx2= int(numpy.floor(0.16*len(thisvz)))
        indx3= int(numpy.floor(0.84*len(thisvz)))      
        medians[ii]= thisvz[indx1]
        smin[ii]= thisvz[indx2]
        splus[ii]= thisvz[indx3]
    rs= numpy.linspace((rs[1]+rs[0])/2.,(rs[-1]+rs[-2])/2.,nbins)
    bovy_plot.bovy_plot(rs,medians,'-',overplot=True,lw=2.,color='0.8')
    indx= numpy.array([not numpy.isnan(splus[ii]) for ii in range(len(splus))],dtype='bool')
    pyplot.fill_between(rs,medians,splus,color='0.5',zorder=-10,where=indx)
    pyplot.fill_between(rs,smin,medians,color='0.5',zorder=-10,where=indx)
    #bovy_plot.bovy_plot(rs,smin,'--',overplot=True,lw=2.,color='0.6')
    #bovy_plot.bovy_plot(rs,splus,'g--',overplot=True,lw=2.,color='0.6')
    if not options.type.lower() == 'datavzz':
        if options.metal.lower() == 'poor':
            txt= r'$\alpha\mathrm{-old}$'
        else:
            txt= r'$\alpha\mathrm{-young}$'
        pyplot.annotate(txt,(0.5,1.06),xycoords='axes fraction',
                        horizontalalignment='center',
                        verticalalignment='top',size=20.)
    bovy_plot.bovy_end_print(os.path.join(args[0],options.type+'_'
                                          +options.sample+'_'+
                                          options.metal+'.'+ext))

def get_options():
    usage = "usage: %prog [options] <savedir>\n\nsavedir= name of the directory that the figures will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--metal",dest='metal',default='rich',
                      help="Use metal-poor or rich sample ('poor', 'rich' or 'all')")
    parser.add_option("-t","--type",dest='type',default='datavzR',
                      help="Type of figure to make ('datavzR')")
    parser.add_option("--png",action="store_true", dest="png",
                      default=False,
                      help="Save as png, otherwise ps")
    return parser

if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    plotVelocityData(options,args)
