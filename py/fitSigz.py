import os, os.path
import numpy
import cPickle as pickle
from optparse import OptionParser
def fitSigz(parser):
    (options,args)= parser.parse_args()
    if len(args) == 0:
        parser.print_help()
        return
    #First read the data
    data= readData(metal=options.metal)

def readData(metal='rich'):
    rawdata= numpy.loadtxt(os.path.join(os.getenv('DATADIR'),'bovy',
                                          'segue-local','gdwarf_raw.dat'))
    #Select sample
    print rawdata.shape
    if metal == 'rich':
        indx= (rawdata[:,16] > -0.4)*(rawdata[:,16] < 0.5)\
            *(rawdata[:,18] > -0.25)*(rawdata[:,18] < 0.2)
    elif metal == 'poor':
        indx= (rawdata[:,16] > -1.5)*(rawdata[:,16] < -0.5)\
            *(rawdata[:,18] > 0.25)*(rawdata[:,18] < 0.5)
    else:
        indx= (rawdata[:,16] > -2.)*(rawdata[:,16] < 0.5)\
            *(rawdata[:,18] > -0.25)*(rawdata[:,18] < 0.5)
    rawdata= rawdata[indx,:]
    print rawdata.shape

def get_options():
    usage = "usage: %prog [options] <savefilename>\n\nsavefilename= name of the file that the fit/samples will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("-o",dest='plotfile',
                      help="Name of file for plot")
    parser.add_option("--metal",dest='metal',
                      help="Use metal-poor or rich sample ('poor', 'rich' or 'all')")
    #parser.add_option("--star",action="store_true", dest="star",
    #                  default=False,
    #                  help="Fit stars")

if __name__ == '__main__':
    fitSigv(get_options())
