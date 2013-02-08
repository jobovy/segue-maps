import os, os.path
from pixelFitDF import get_options
from plotDensComparisonDFMulti import plotDensComparisonDFMulti
from plotVelComparisonDFMulti import plotVelComparisonDFMulti
def plotAllComparisonDFMulti(options,args):
    #First do all dens comparisons
    if not options.justvel:
        options.type= 'z'
        plotDensComparisonDFMulti(options,args)
        #options.type= 'R'
        #plotDensComparisonDFMulti(options,args)
    options.type= 'vt'
    plotVelComparisonDFMulti(options,args)
    options.type= 'vz'
    plotVelComparisonDFMulti(options,args)
    options.type= 'vr'
    plotVelComparisonDFMulti(options,args)
    return None

if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    try:
        os.mkdir(os.path.join('..',args[0][0:-1]))
    except OSError:
        pass
    args[0]= os.path.join('..',args[0][0:-1],args[0])
    if options.allgroups:
        options.group= 'aenhanced'
        plotAllComparisonDFMulti(options,args)
        options.group= 'apoor'
        plotAllComparisonDFMulti(options,args)
        options.group= 'apoorfpoor'
        plotAllComparisonDFMulti(options,args)
        options.group= 'aintermediate'
        plotAllComparisonDFMulti(options,args)
    else:
        plotAllComparisonDFMulti(options,args)
