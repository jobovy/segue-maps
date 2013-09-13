import os, os.path
import sys
import numpy
from scipy import optimize
import pickle
from galpy.util import bovy_plot
from matplotlib import pyplot, cm
from selectFigs import _squeeze
from plotOverview import expcurve
_INCLUDEDR= True
_CSV= False
def tableSurf(savefilename,outfilename):
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
    fehs= fehs[indx]
    afes= afes[indx]
    #Now table
    cmdline= '%python tableSurf.py '+savefilename+' '+outfilename
    outfile= open(outfilename,'w')
    delimiter= ' & '
    if _CSV: delimiter= ','
    for ii in range(len(fehs)):
        printline= '%.2f' % fehs[ii]
        printline+= delimiter
        printline+= '%.3f' % afes[ii]
        printline+= delimiter
        printline+= '%.2f' % surfrs[ii]
        printline+= delimiter
        printline+= '%.1f' % surfs[ii]
        printline+= delimiter
        printline+= '%.1f' % surferrs[ii]
        printline+= delimiter
        printline+= '%.2f' % (8.-surfrs[ii])
        printline+= delimiter
        printline+= '%.1f' % kzs[ii]
        printline+= delimiter
        printline+= '%.1f' % kzerrs[ii]
        if not _CSV:
            printline+= '\\\\'
        outfile.write(printline+'\n')
    if not _CSV:
        outfile.write('\\enddata\n')
        outfile.write(cmdline+'\n')
    outfile.close()
    return None   

if __name__ == '__main__':
    tableSurf(sys.argv[1],sys.argv[2])
