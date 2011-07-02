import os, os.path
import numpy
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_coords
_VERBOSE=True
_DEBUG=True
def fitSigz(parser):
    (options,args)= parser.parse_args()
    if len(args) == 0:
        parser.print_help()
        return
    #First read the data
    if _VERBOSE:
        print "Reading and parsing data ..."
    XYZ,vxvyvz,cov_vxvyvz,rawdata= readData(metal=options.metal)
    return None

def readData(metal='rich'):
    rawdata= numpy.loadtxt(os.path.join(os.getenv('DATADIR'),'bovy',
                                          'segue-local','gdwarf_raw.dat'))
    #Select sample
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
    ndata= len(rawdata[:,0])
    #calculate distances and velocities
    lb= bovy_coords.radec_to_lb(rawdata[:,0],rawdata[:,1],degree=True)
    XYZ= bovy_coords.lbd_to_XYZ(lb[:,0],lb[:,1],rawdata[:,26]/1000.,
                                degree=True)
    pmllpmbb= bovy_coords.pmrapmdec_to_pmllpmbb(rawdata[:,22],rawdata[:,24],
                                                rawdata[:,0],rawdata[:,1],
                                                degree=True)
    vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(rawdata[:,20],pmllpmbb[:,0],
                                             pmllpmbb[:,1],lb[:,0],lb[:,1],
                                             rawdata[:,26]/1000.,degree=True)
    #Solar motion
    vxvyvz[:,0]+= -11.1
    vxvyvz[:,1]+= 12.24
    vxvyvz[:,2]+= 7.25
    #print numpy.mean(vxvyvz[:,2]), numpy.std(vxvyvz[:,2])
    #Propagate uncertainties
    cov_pmradec= numpy.zeros((ndata,2,2))
    cov_pmradec[:,0,0]= rawdata[:,23]**2.
    cov_pmradec[:,1,1]= rawdata[:,25]**2.
    cov_pmllbb= bovy_coords.cov_pmrapmdec_to_pmllpmbb(cov_pmradec,rawdata[:,0],
                                                      rawdata[:,1],degree=True)
    cov_vxvyvz= bovy_coords.cov_dvrpmllbb_to_vxyz(rawdata[:,26]/1000.,
                                                  rawdata[:,27]/1000.,
                                                  rawdata[:,21],
                                                  pmllpmbb[:,0],pmllpmbb[:,1],
                                                  cov_pmllbb,lb[:,0],lb[:,1],
                                                  degree=True)
    #print numpy.median(cov_vxvyvz[:,0,0]), numpy.median(cov_vxvyvz[:,1,1]), numpy.median(cov_vxvyvz[:,2,2])
    if _DEBUG:
        #Compare U with Chao Liu
        print "Z comparison with Chao Liu"
        print numpy.mean(XYZ[:,2]-rawdata[:,31]/1000.)#mine are in kpc
        print numpy.std(XYZ[:,2]-rawdata[:,31]/1000.)
        #Compare U with Chao Liu
        print "U comparison with Chao Liu"
        print numpy.mean(vxvyvz[:,0]-rawdata[:,32])
        print numpy.std(vxvyvz[:,0]-rawdata[:,32])
        print numpy.mean(numpy.sqrt(cov_vxvyvz[:,0,0])-rawdata[:,33])
        print numpy.std(numpy.sqrt(cov_vxvyvz[:,0,0])-rawdata[:,33])
        #V
        print "V comparison with Chao Liu"
        print numpy.mean(vxvyvz[:,1]-rawdata[:,34])
        print numpy.std(vxvyvz[:,1]-rawdata[:,34])
        print numpy.mean(numpy.sqrt(cov_vxvyvz[:,1,1])-rawdata[:,35])
        print numpy.std(numpy.sqrt(cov_vxvyvz[:,1,1])-rawdata[:,35])
        #W
        print "W comparison with Chao Liu"
        print numpy.mean(vxvyvz[:,2]-rawdata[:,36])
        print numpy.std(vxvyvz[:,2]-rawdata[:,36])
        print numpy.mean(numpy.sqrt(cov_vxvyvz[:,2,2])-rawdata[:,37])
        print numpy.std(numpy.sqrt(cov_vxvyvz[:,2,2])-rawdata[:,37])
    #Load for output
    return (XYZ,vxvyvz,cov_vxvyvz,rawdata)
    
def get_options():
    usage = "usage: %prog [options] <savefilename>\n\nsavefilename= name of the file that the fit/samples will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("-o",dest='plotfile',
                      help="Name of file for plot")
    parser.add_option("--metal",dest='metal',default='rich',
                      help="Use metal-poor or rich sample ('poor', 'rich' or 'all')")
    #parser.add_option("--star",action="store_true", dest="star",
    #                  default=False,
    #                  help="Fit stars")
    return parser

if __name__ == '__main__':
    fitSigz(get_options())
