import os, os.path
import sys
import numpy
import cPickle as pickle
from optparse import OptionParser
from galpy.potential import LogarithmicHaloPotential, MWPotential
from galpy.orbit import Orbit
from segueSelect import _append_field_recarray, _ERASESTR
from fitSigz import readData, _APOORFEHRANGE, _ARICHFEHRANGE, _ZSUN
_MAXT= 100.
def calcOrbits(parser):
    options,args= parser.parse_args()
    #Read data
    XYZ,vxvyvz,cov_vxvyvz,rawdata= readData(metal='allall',
                                            sample=options.sample,
                                            loggmin=False,
                                            snmin=False,
                                            select=options.select)
    #Define potential
    if options.logp:
        pot= LogarithmicHaloPotential(normalize=1.)
    else:
        pot= MWPotential
    ts= numpy.linspace(0.,_MAXT,10000) #times to integrate
    if os.path.exists(args[0]):#Load savefile
        savefile= open(args[0],'rb')
        orbits= pickle.load(savefile)
        _ORBITSLOADED= True
        try:
            samples= pickle.load(savefile)
        except EORError:
            _SAMPLESLOADED= False
        else:
            _SAMPLESLOADED= True
        finally:
            savefile.close()
    else:
        _ORBITSLOADED= False
    if not _ORBITSLOADED:
        #First calculate orbits
        es, rmeans, rperis, raps, zmaxs = [], [], [], [], []
        for ii in range(len(rawdata)):
            sys.stdout.write('\r'+"Working on object %i/%i" % (ii,len(rawdata)))
            sys.stdout.flush()
            #Integrate the orbit
            data= rawdata[ii]
            o= Orbit([data.ra,data.dec,data.dist,data.pmra,data.pmdec,data.vr],
                     radec=True,vo=220.,ro=8.,zo=_ZSUN)
            o.integrate(ts,pot)
            es.append(o.e())
            rperis.append(o.rperi())
            raps.append(o.rap())
            zmaxs.append(o.zmax())
            rmeans.append(0.5*(o.rperi()+o.rap()))
        sys.stdout.write('\r'+_ERASESTR+'\r')
        sys.stdout.flush()
        es= numpy.array(es)
        rmeans= numpy.array(rmeans)
        rperis= numpy.array(rperis)
        raps= numpy.array(raps)
        zmaxs= numpy.array(zmaxs)
        orbits= _append_field_recarray(rawdata,'e',es)
        orbits= _append_field_recarray(rawdata,'rmean',rmeans)
        orbits= _append_field_recarray(rawdata,'rmean',rmeans)
        orbits= _append_field_recarray(rawdata,'rperi',rperis)
        orbits= _append_field_recarray(rawdata,'rap',raps)
        orbits= _append_field_recarray(rawdata,'zmax',zmaxs)
        #Pickle
        savefile= open(args[0],'wb')
        pickle.dump(orbits,savefile)
        savefile.close()
    return None

def get_options():
    usage = "usage: %prog [options] <savedir>\n\nsavedir= name of the file that the orbits will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--select",dest='select',default='all',
                      help="Select 'all' or 'program' stars")
    parser.add_option("--logp",action="store_true", dest="logp",
                      default=False,
                      help="Use a logarithmic potential rather than the mp,np,hp MWPotential")
    return parser

if __name__ == '__main__':
    calcOrbits(get_options())


