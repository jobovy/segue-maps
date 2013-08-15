import cPickle as pickle
import numpy
from galpy.util import bovy_plot
import monoAbundanceMW
from pixelFitDF import get_options, approxFitResult
def plotQdfPreviousFits(options,args):
    #Get the bins
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    savefile.close()
    #Run through them and get the apprimxate results
    lnhrin= numpy.empty(len(fehs))
    lnsrin= numpy.empty(len(fehs))
    lnszin= numpy.empty(len(fehs))
    for ii in range(len(fehs)):
        out= approxFitResult(fehs[ii],afes[ii])
        lnhrin[ii]= out[0]
        lnsrin[ii]= out[1]
        lnszin[ii]= out[2]
    #Now plot: hR
    bovy_plot.bovy_print()
    monoAbundanceMW.plotPixelFunc(fehs,afes,numpy.exp(lnhrin)*8.,
                                  vmin=1.5,vmax=4.86,
                                  zlabel=r'$\mathrm{Input\ radial\ scale\ length\ (kpc)}$')
    #Plotname
    spl= options.outfilename.split('.')
    newname= ''
    for jj in range(len(spl)-1):
        newname+= spl[jj]
        if not jj == len(spl)-2: newname+= '.'
    newname+= '_hrin.'
    newname+= spl[-1]
    bovy_plot.bovy_end_print(newname)
    #Now plot: sR
    bovy_plot.bovy_print()
    monoAbundanceMW.plotPixelFunc(fehs,afes,numpy.exp(lnsrin)*220.,
                                  vmin=30.,vmax=60.,
                                  zlabel=r'$\mathrm{Input\ radial\ velocity\ dispersion\ (km\,s}^{-1})$')
    #Plotname
    spl= options.outfilename.split('.')
    newname= ''
    for jj in range(len(spl)-1):
        newname+= spl[jj]
        if not jj == len(spl)-2: newname+= '.'
    newname+= '_srin.'
    newname+= spl[-1]
    bovy_plot.bovy_end_print(newname)
    #Now plot: sZ
    bovy_plot.bovy_print()
    monoAbundanceMW.plotPixelFunc(fehs,afes,numpy.exp(lnszin)*220.,
                                  vmin=10.,vmax=80.,
                                  zlabel=r'$\mathrm{Input\ vertical\ velocity\ dispersion\ (km\,s}^{-1})$')
    #Plotname
    spl= options.outfilename.split('.')
    newname= ''
    for jj in range(len(spl)-1):
        newname+= spl[jj]
        if not jj == len(spl)-2: newname+= '.'
    newname+= '_szin.'
    newname+= spl[-1]
    bovy_plot.bovy_end_print(newname)

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    plotQdfPreviousFits(options,args)
