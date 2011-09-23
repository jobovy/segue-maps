import os, os.path
import sys
import math
import numpy
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_coords, bovy_plot, save_pickles
from matplotlib import pyplot, cm
from segueSelect import read_gdwarfs, read_kdwarfs, _gi_gr, _mr_gi, \
    segueSelect, _GDWARFFILE, _KDWARFFILE
from selectFigs import _squeeze
from fitDensz import _TwoDblExpDensity, _HWRLikeMinus, _ZSUN, DistSpline, \
    _ivezic_dist, _NDS, cb, _HWRDensity, _HWRLike
from pixelFitDens import pixelAfeFeh
from predictBellHalo import predictDiskMass
_NGR= 11
_NFEH=11
def calcMass(options,args):
    if options.sample.lower() == 'g':
        if options.select.lower() == 'program':
            raw= read_gdwarfs(_GDWARFFILE,logg=True,ebv=True,sn=True)
        else:
            raw= read_gdwarfs(logg=True,ebv=True,sn=True)
    elif options.sample.lower() == 'k':
        if options.select.lower() == 'program':
            raw= read_kdwarfs(_KDWARFFILE,logg=True,ebv=True,sn=True)
        else:
            raw= read_kdwarfs(logg=True,ebv=True,sn=True)
    #Bin the data
    binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
    #Savefile
    if os.path.exists(args[0]):#Load savefile
        savefile= open(args[0],'rb')
        mass= pickle.load(savefile)
        ii= pickle.load(savefile)
        jj= pickle.load(savefile)
        savefile.close()
    else:
        fits= []
        ii, jj= 0, 0
    #parameters
    if os.path.exists(args[1]):#Load initial
        savefile= open(args[1],'rb')
        fits= pickle.load(savefile)
        savefile.close()
    else:
        print "Error: must provide parameters of best fits"
        print "Returning ..."
        return None
    #Set up model etc.
    if options.model.lower() == 'hwr':
        densfunc= _HWRDensity
    elif options.model.lower() == 'twodblexp':
        densfunc= _TwoDblExpDensity
    like_func= _HWRLikeMinus
    pdf_func= _HWRLike
    if options.sample.lower() == 'g':
        colorrange=[0.48,0.55]
    elif options.sample.lower() == 'k':
        colorrange=[0.55,0.75]
    #Load selection function
    plates= numpy.array(list(set(list(raw.plate))),dtype='int') #Only load plates that we use
    print "Using %i plates, %i stars ..." %(len(plates),len(raw))
    sf= segueSelect(plates=plates,type_faint='tanhrcut',
                    sample=options.sample,type_bright='tanhrcut',
                    sn=True,select=options.select)
    platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                     degree=True)
    indx= [not 'faint' in name for name in sf.platestr.programname]
    platebright= numpy.array(indx,dtype='bool')
    indx= ['faint' in name for name in sf.platestr.programname]
    platefaint= numpy.array(indx,dtype='bool')
    if options.sample.lower() == 'g':
        grmin, grmax= 0.48, 0.55
        rmin,rmax= 14.5, 20.2
    #Run through the bins
    mass= []
    while ii < len(binned.fehedges)-1:
        while jj < len(binned.afeedges)-1:
            data= binned(binned.feh(ii),binned.afe(jj))
            if len(data) < options.minndata:
                mass.append(None)
                jj+= 1
                if jj == len(binned.afeedges)-1: 
                    jj= 0
                    ii+= 1
                    break
                continue               
            print binned.feh(ii), binned.afe(jj), len(data)
            fehindx= binned.fehindx(binned.feh(ii))
            afeindx= binned.afeindx(binned.afe(jj))
            thisparams= fits[afeindx+fehindx*binned.npixafe()]
            #set up feh and color
            feh= binned.feh(ii)
            fehrange= [binned.fehedges[ii],binned.fehedges[ii+1]]
            #FeH
            fehdist= DistSpline(*numpy.histogram(data.feh,bins=5,
                                                 range=fehrange),
                                 xrange=fehrange,dontcuttorange=False)
            #Color
            colordist= DistSpline(*numpy.histogram(data.dered_g\
                                                       -data.dered_r,
                                                   bins=9,range=colorrange),
                                   xrange=colorrange)
            
            #Age marginalization
            afe= binned.afe(jj)
            if afe > 0.25: agemin, agemax= 7.,10.
            else: agemin,agemax= 1.,8.
            mass.append(predictDiskMass(densfunc,
                                        thisparams,sf,colordist,fehdist,
                                        fehrange[0],fehrange[1],feh,
                                        data,colorrange[0],colorrange[1],
                                        agemin,agemax,
                                        normalize=options.normalize))
            print mass[-1]
            jj+= 1
            if jj == len(binned.afeedges)-1: 
                jj= 0
                ii+= 1
            save_pickles(args[0],mass,ii,jj)
            if jj == 0: #this means we've reset the counter 
                break
    save_pickles(args[0],mass,ii,jj)
    return None

def get_options():
    usage = "usage: %prog [options] <savefile> <savefile>\n\nsavefile= name of the file that the mass will be saved to\nsavefile = name of the file that has the best fits"
    parser = OptionParser(usage=usage)
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--select",dest='select',default='all',
                      help="Select 'all' or 'program' stars")
    parser.add_option("--dfeh",dest='dfeh',default=0.05,type='float',
                      help="FeH bin size")   
    parser.add_option("--dafe",dest='dafe',default=0.05,type='float',
                      help="[a/Fe] bin size")   
    parser.add_option("--minndata",dest='minndata',default=100,type='int',
                      help="Minimum number of objects in a bin to perform a fit")   
    parser.add_option("--model",dest='model',default='twodblexp',
                      help="Model to fit")
    parser.add_option("-o","--plotfile",dest='plotfile',default=None,
                      help="Name of the file for plot")
    parser.add_option("-t","--type",dest='type',default='hr',
                      help="Quantity to plot ('hz', 'hr', 'afe', 'feh'")
    parser.add_option("--normalize",dest='normalize',default='Z',
                      help="Normalize over Z or over space")
    parser.add_option("--plot",action="store_true", dest="plot",
                      default=False,
                      help="If set, plot, otherwise, fit")
    parser.add_option("--tighten",action="store_true", dest="tighten",
                      default=False,
                      help="If set, tighten axes")
    parser.add_option("--ploterrors",action="store_true", dest="ploterrors",
                      default=False,
                      help="If set, plot the errorbars")
    return parser
  
if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    if options.plot:
        plotPixelFit(options,args)
    else:
        calcMass(options,args)

