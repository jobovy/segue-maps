import numpy
from galpy.util import bovy_plot
import monoAbundanceMW
import AnDistance
from pixelFitDens import pixelAfeFeh
from pixelFitDF import get_options, read_rawdata
def plotAnDistance(args,options):
    raw= read_rawdata(options)
    binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
    #Get fehs from monoAb
    fehs= monoAbundanceMW.fehs(k=(options.sample.lower() == 'k'))
    afes= monoAbundanceMW.afes(k=(options.sample.lower() == 'k'))
    plotthis= numpy.zeros_like(fehs)
    for ii in range(len(fehs)):
        #Get the relevant data
        data= binned(fehs[ii],afes[ii])
        plotthis[ii]= AnDistance.AnDistance(data.dered_g-data.dered_r,
                                            data.feh)
    #Now plot
    bovy_plot.bovy_print()
    monoAbundanceMW.plotPixelFunc(fehs,afes,plotthis,
                                  zlabel=r'$f_\mathrm{distance}$')
    bovy_plot.bovy_end_print(options.outfilename)
    return None

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    plotAnDistance(args,options)
