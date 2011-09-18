import os, os.path
import sys
import math
import numpy
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_coords, bovy_plot, save_pickles
from matplotlib import pyplot, cm
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE
from pixelFitDens import pixelAfeFeh
from fitSigz import _ZSUN
def plotsz2hz(options,args):
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
    if options.tighten:
        tightbinned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe,
                                 fehmin=-2.,fehmax=0.3,afemin=0.,afemax=0.45)
    else:
        tightbinned= binned
    #Savefile1
    if os.path.exists(args[0]):#Load savefile
        savefile= open(args[0],'rb')
        velfits= pickle.load(savefile)
        savefile.close()
    if os.path.exists(args[1]):#Load savefile
        savefile= open(args[1],'rb')
        densfits= pickle.load(savefile)
        savefile.close()
    #Uncertainties are in savefile3 and 4
    if len(args) > 3 and os.path.exists(args[3]):
        savefile= open(args[3],'rb')
        denssamples= pickle.load(savefile)
        savefile.close()
        denserrors= True
    else:
        denssamples= None
        denserrors= False
    if len(args) > 2 and os.path.exists(args[2]):
        savefile= open(args[2],'rb')
        velsamples= pickle.load(savefile)
        savefile.close()
        velerrors= True
    else:
        velsamples= None            
        velerrors= False

def get_options():
    usage = "usage: %prog [options] <savefile1> <savefile2>\n\nsavefile1= name of the file that holds the single disk fits\nsavefile2 = name of the file that holds the double disk fits\nsavefile3= name of the file that has the samplings for the single disk\nsavefile4= name of the file that has the samplings for the double disk fits"
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
    parser.add_option("-o","--plotfile",dest='plotfile',default=None,
                      help="Name of the file for plot")
    parser.add_option("-t","--type",dest='type',default='sz',
                      help="Quantity to plot ('hz' or 'hr')")
    parser.add_option("--subtype",dest='subtype',default='mz',
                      help="Sub-type of plot: when plotting afe, feh, afefeh, or fehafe, plot this")
    parser.add_option("--tighten",action="store_true", dest="tighten",
                      default=False,
                      help="If set, tighten axes")
    return parser

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    plotOneDiskVsTwoDisks(options,args)

