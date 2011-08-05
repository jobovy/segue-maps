import os, os.path
import numpy
from optparse import OptionParser
from galpy.util import bovy_plot, bovy_coords
from matplotlib import pyplot
import segueSelect
from fitSigz import readData, _APOORFEHRANGE, _ARICHFEHRANGE
from fitDensz import FeHXDDist, DistSpline
from compareDataModel import _add_coordinset
def plotMetallicityColor(options,args):
    if options.png: ext= 'png'
    else: ext= 'ps'
    #Load data
    XYZ,vxvyvz,cov_vxvyvz,data= readData(metal=options.metal,
                                         sample=options.sample)
    #Load plates
    platestr= segueSelect._load_fits(os.path.join(segueSelect._SEGUESELECTDIR,
                                      'segueplates.fits'))
    #Set up ranges
    if options.sample.lower() == 'g':
        xrange=[0.46,0.57]
        rmin, rmax= 14.5, 20.2
        grmin, grmax= 0.48, 0.55
        rx= 0.47
        colorrange=[0.48,0.55]
    elif options.sample.lower() == 'k':
        xrange=[0.51,0.79]
        rmin, rmax= 14.5, 19.
        grmin, grmax= 0.55,0.75
        rx= 0.01/.11*(xrange[1]-xrange[0])+xrange[0]
        colorrange=[0.55,0.75]
    if options.metal.lower() == 'rich':
        yrange=[-0.55,0.5]
        ry= 0.275
        fehrange= _APOORFEHRANGE
    elif options.metal.lower() == 'poor':
        yrange=[-1.6,0.3]
        ry= yrange[1]-(.5-0.275)*(yrange[1]-yrange[0])
        fehrange= _ARICHFEHRANGE
    #First plot all data, faint, and bright
    bovy_plot.bovy_print()
    _plotMC_single(data,options,args,all=True,overplot=False,xrange=xrange,
                   yrange=yrange,fehrange=fehrange,colorrange=colorrange)
    _overplot_model(data,xrange=xrange,yrange=yrange,fehrange=fehrange,
                    colorrange=colorrange)
    bovy_plot.bovy_end_print(os.path.join(args[0],'FeH_gr_'+options.sample+'_'+options.metal+'.'+ext))
    #Then plot them pixel-by-pixel
    platelb= bovy_coords.radec_to_lb(platestr.ra,platestr.dec,
                                     degree=True)
    nside= 1
    for ii in range(12*nside**2):
        #First find all of the plates in this pixel
        indx= (platestr.healpix_level1 == ii)
        if numpy.sum(indx) == 0: continue #No match
        thisplatestr= platestr[indx]
        platels= platelb[indx,0]
        platebs= platelb[indx,1]
        indx= []
        for jj in range(len(data.ra)):
            if data[jj].plate in list(thisplatestr.plate):
                indx.append(True)
            else:
                indx.append(False)
        indx= numpy.array(indx,dtype='bool')
        if numpy.sum(indx) == 0: continue #No matches
        bovy_plot.bovy_print()
        _plotMC_single(data[indx],options,args,all=False,overplot=False,
                       xrange=xrange,yrange=yrange,platels=platels,
                       platebs=platebs,
                       ry=ry,rx=rx,
                       rmin=rmin,rmax=rmax,grmin=grmin,grmax=grmax)
        _overplot_model(data,xrange=xrange,yrange=yrange,fehrange=fehrange,
                        colorrange=colorrange)
        bovy_plot.bovy_end_print(os.path.join(args[0],'FeH_gr_'+options.sample+'_'+options.metal+'_%i.' % ii + ext))
    return None

def _overplot_model(data,xrange=None,yrange=None,fehrange=None,
                    colorrange=None):
    #Load model distributions
    #FeH
    fehdist= DistSpline(*numpy.histogram(data.feh,bins=11,range=fehrange),
                         xrange=fehrange)
    nfehs= 1001
    fehs= numpy.linspace(yrange[0],yrange[1],nfehs)
    mfehs= numpy.zeros(nfehs)
    for ii in range(nfehs):
        mfehs[ii]= fehdist(fehs[ii])
    mfehs/= numpy.nansum(mfehs)*(fehs[1]-fehs[0])    
    #Color
    cdist= DistSpline(*numpy.histogram(data.dered_g-data.dered_r,
                                       bins=9,range=colorrange),
                       xrange=colorrange)
    ncs= 1001
    cs= numpy.linspace(xrange[0],xrange[1],ncs)
    mcs= numpy.zeros(nfehs)
    for ii in range(nfehs):
        mcs[ii]= cdist(cs[ii])
    mcs/= numpy.nansum(mcs)*(cs[1]-cs[0])    
    #Overplot model FeH
    from matplotlib.ticker import NullFormatter
    fig= pyplot.gcf()
    nullfmt   = NullFormatter()         # no labels
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    axScatter = pyplot.axes(rect_scatter)
    axHistx = pyplot.axes(rect_histx)
    axHisty = pyplot.axes(rect_histy)
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHistx.yaxis.set_major_formatter(nullfmt)
    axHisty.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    fig.sca(axHisty)
    bovy_plot.bovy_plot(mfehs,fehs,'k-',overplot=True)
    fig.sca(axHistx)
    bovy_plot.bovy_plot(cs,mcs,'k-',overplot=True)
    return None

def _plotMC_single(data,options,args,all=False,overplot=False,xrange=None,
                   yrange=None,platels=None,platebs=None,
                   rmin=None,rmax=None,grmin=None,grmax=None,
                   rx=None,ry=None,fehrange=None,colorrange=None):
    if all:
        bovy_plot.scatterplot(data.dered_g-data.dered_r,
                              data.feh,
                              'k,',
                              bins=21,
                              xrange=xrange,
                              yrange=yrange,
                              xlabel=r'$g-r\ [\mathrm{mag}]$',
                              ylabel=r'$[\mathrm{Fe/H}]$',
                              onedhists=True)
        platestr= '\mathrm{all\ plates}'
        bovy_plot.bovy_text(r'$'+platestr+'$'
                            +'\n'+
                            '$%i \ \ \mathrm{stars}$' % 
                            len(data.feh),top_right=True,size=16)
    else:
        bovy_plot.bovy_plot(data.dered_g-data.dered_r,
                              data.feh,
                              'k,',
                              bins=21,
                              xrange=xrange,
                              yrange=yrange,
                              xlabel=r'$g-r\ [\mathrm{mag}]$',
                              ylabel=r'$[\mathrm{Fe/H}]$',
                              onedhists=True)
        platestr= '%i\ \mathrm{plates}' % len(set(list(data.plate)))
        lbstr= '$l = %i^\circ \pm %i^\circ$' % (
            int(numpy.mean(platels)),int(numpy.std(platels)))+'\n'\
            +'$b = %i^\circ\pm%i^\circ$' % (int(numpy.mean(platebs)),
                                            int(numpy.std(platebs)))
        bovy_plot.bovy_text(r'$'+platestr+'$'
                            +'\n'+
                            '$%i \ \ \mathrm{stars}$' % 
                            len(data.feh)
                            +'\n'+
                            lbstr,top_right=True,size=16)
        _add_coordinset(rx=rx,ry=ry,platels=platels,platebs=platebs,
                        feh=numpy.mean(data.feh),
                        rmin=rmin,rmax=rmax,
                        grmin=grmin,grmax=grmin)

def plotDMMetallicityColor(options,args):
    """Make a density plot of DM vs FeH and g-r"""
    if options.png: ext= 'png'
    else: ext= 'ps'
    if options.metal.lower() == 'rich':
        yrange=[-0.55,0.5]
        fehrange= _APOORFEHRANGE
    elif options.metal.lower() == 'poor':
        yrange=[-1.6,0.3]
        fehrange= _ARICHFEHRANGE
    xrange=[0.46,0.57]
    grmin, grmax= 0.48, 0.55
    colorrange=[0.48,0.55]
    #Set up arrays
    nfehs, ngrs= 201,201
    grs= numpy.linspace(xrange[0],xrange[1],nfehs)
    fehs= numpy.linspace(yrange[0],yrange[1],ngrs)
    plotthis= numpy.zeros((ngrs,nfehs))
    for ii in range(ngrs):
        for jj in range(nfehs):
            if grs[ii] < colorrange[0] \
                    or grs[ii] > colorrange[1] \
                    or fehs[jj] < fehrange[0] \
                    or fehs[jj] > fehrange[1]:
                plotthis[ii,jj]= numpy.nan
                continue
            plotthis[ii,jj]= segueSelect._mr_gi(segueSelect._gi_gr(grs[ii]),fehs[jj])
    if options.sample.lower() == 'g':
        if options.metal.lower() == 'rich':
            levels= [4.5,4.75,5.,5.25,5.5]
        else:
            levels= [5.25,5.5,5.75,6.,6.25]
    cntrlabelcolors= ['w' for ii in range(3)]
    cntrlabelcolors.extend(['k' for ii in range(2)])
    #nlevels= 6
    #levelsstart= int(20.*numpy.nanmin(plotthis))/20.
    #levelsd= int(20.*(numpy.nanmax(plotthis)-numpy.nanmin(plotthis)))/20.
    #levels= [levelsstart+ii/float(nlevels)*levelsd for ii in range(nlevels)]
    #Plot it
    bovy_plot.bovy_print()
    bovy_plot.bovy_dens2d(plotthis.T,origin='lower',
                          xlabel=r'$g-r\ [\mathrm{mag}]$',
                          ylabel=r'$[\mathrm{Fe/H}]$',
                          zlabel=r'$M_r\ [\mathrm{mag}]$',
                          colorbar=True,
                          cmap=pyplot.cm.gist_gray,
                          contours=True,
                          levels=levels,
                          cntrcolors=cntrlabelcolors,
                          cntrlabel=True,
                          cntrlabelcolors=cntrlabelcolors,
                          cntrinline=True,
                          interpolation='nearest',
                          extent=[xrange[0],xrange[1],
                                  yrange[0],yrange[1]],
                          aspect=(xrange[1]-xrange[0])/\
                              (yrange[1]-yrange[0]),
                          shrink=.78)
    bovy_plot.bovy_end_print(os.path.join(args[0],'dm_FeH_gr_'+options.sample+'_'+options.metal+'.'+ext))

def get_options():
    usage = "usage: %prog [options] <savedir>\n\nsavedir= name of the directory that the comparisons will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--metal",dest='metal',default='rich',
                      help="Use metal-poor or rich sample ('poor', 'rich' or 'all')")
    parser.add_option("--plottype",dest='plottype',default='fehcolor',
                      help="Type of plot to make ('fehcolor,dmfehcolor')")
    parser.add_option("--png",action="store_true", dest="png",
                      default=False,
                      help="Save as png, otherwise ps")
    return parser

if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    if options.plottype == 'fehcolor':
        plotMetallicityColor(options,args)
    else:
        plotDMMetallicityColor(options,args)
