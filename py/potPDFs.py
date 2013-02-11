import os, os.path
import sys
import copy
import tempfile
import math
import numpy
from scipy import optimize, interpolate, linalg, integrate
from scipy.maxentropy import logsumexp
import cPickle as pickle
from optparse import OptionParser
import multi
import multiprocessing
import monoAbundanceMW
from galpy.util import bovy_coords, bovy_plot, save_pickles
from galpy import potential
from galpy.df import quasiisothermaldf
from galpy.util import bovy_plot
import monoAbundanceMW
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE, \
    segueSelect, _mr_gi, _gi_gr, _ERASESTR, _append_field_recarray, \
    ivezic_dist_gr
from fitDensz import cb, _ZSUN, DistSpline, _ivezic_dist, _NDS
from pixelFitDens import pixelAfeFeh
from pixelFitDF import _REFV0, get_options, read_rawdata, get_potparams, \
    get_dfparams, _REFR0, get_vo, get_ro, setup_potential, setup_aA
class potPDFs:
    """Class for representing potential PDFs"""
    def __init__(self,options,args):
        """Initialize"""
        raw= read_rawdata(options)
        #Bin the data
        binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
        tightbinned= binned
        #Map the bins with ndata > minndata in 1D
        fehs, afes= [], []
        counter= 0
        abindx= numpy.zeros((len(binned.fehedges)-1,len(binned.afeedges)-1),
                            dtype='int')
        for ii in range(len(binned.fehedges)-1):
            for jj in range(len(binned.afeedges)-1):
                data= binned(binned.feh(ii),binned.afe(jj))
                if len(data) < options.minndata:
                    continue
                #print binned.feh(ii), binned.afe(jj), len(data)
                fehs.append(binned.feh(ii))
                afes.append(binned.afe(jj))
                abindx[ii,jj]= counter
                counter+= 1
        nabundancebins= len(fehs)
        fehs= numpy.array(fehs)
        afes= numpy.array(afes)
        #Load each of the solutions
        sols= []
        chi2s= []
        samples= []
        savename= args[0]
        initname= options.init
        for ii in range(nabundancebins):
            spl= savename.split('.')
            newname= ''
            for jj in range(len(spl)-1):
                newname+= spl[jj]
                if not jj == len(spl)-2: newname+= '.'
            newname+= '_%i.' % ii
            newname+= spl[-1]
            savefilename= newname
            #Read savefile
            try:
                savefile= open(savefilename,'rb')
            except IOError:
                print "WARNING: MISSING ABUNDANCE BIN"
                sols.append(None)
                chi2s.append(None)
                samples.append(None)
            else:
                sols.append(pickle.load(savefile))
                chi2s.append(pickle.load(savefile))
                samples.append(pickle.load(savefile))
                savefile.close()
        mapfehs= monoAbundanceMW.fehs()
        mapafes= monoAbundanceMW.afes()
        #Calculate everything
        fehs= []
        afes= []
        ndatas= []
        #Basic parameters
        rds= []

        
