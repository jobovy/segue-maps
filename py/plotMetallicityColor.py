import os, os.path
import numpy
from optparse import OptionParser
from galpy.util import bovy_plot, bovy_coords
import segueSelect
from fitSigz import readData
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
    elif options.sample.lower() == 'k':
        xrange=[0.51,0.79]
        rmin, rmax= 14.5, 19.
        grmin, grmax= 0.55,0.75
        rx= 0.01/.11*(xrange[1]-xrange[0])+xrange[0]
    if options.metal.lower() == 'rich':
        yrange=[-0.5,0.5]
        ry= 0.275
    elif options.metal.lower() == 'poor':
        yrange=[-1.6,0.3]
        ry= yrange[1]-(.5-0.275)*(yrange[1]-yrange[0])
    #First plot all data
    bovy_plot.bovy_print()
    _plotMC_single(data,options,args,all=True,overplot=False,xrange=xrange,
                   yrange=yrange)
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
        bovy_plot.bovy_end_print(os.path.join(args[0],'FeH_gr_'+options.sample+'_'+options.metal+'_%i.' % ii + ext))
    return None

def _plotMC_single(data,options,args,all=False,overplot=False,xrange=None,
                   yrange=None,platels=None,platebs=None,
                   rmin=None,rmax=None,grmin=None,grmax=None,
                   rx=None,ry=None):
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
    if all:
        platestr= '\mathrm{all\ plates}'
        bovy_plot.bovy_text(r'$'+platestr+'$'
                            +'\n'+
                            '$%i \ \ \mathrm{stars}$' % 
                            len(data.feh),top_right=True)
    else:
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
                            lbstr,top_right=True)
        _add_coordinset(rx=rx,ry=ry,platels=platels,platebs=platebs,
                        feh=numpy.mean(data.feh),
                        rmin=rmin,rmax=rmax,
                        grmin=grmin,grmax=grmin)

def get_options():
    usage = "usage: %prog [options] <savedir>\n\nsavedir= name of the directory that the comparisons will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--metal",dest='metal',default='rich',
                      help="Use metal-poor or rich sample ('poor', 'rich' or 'all')")
    parser.add_option("--png",action="store_true", dest="png",
                      default=False,
                      help="Save as png, otherwise ps")
    return parser

if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    plotMetallicityColor(options,args)
