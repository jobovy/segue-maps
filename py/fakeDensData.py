import sys
import os, os.path
import math
import numpy
import cPickle as pickle
from matplotlib import pyplot
from optparse import OptionParser
from galpy.util import bovy_coords
import segueSelect
from fitDensz import _HWRDensity, _FlareDensity, _TiedFlareDensity, \
    _TwoVerticalDensity, _const_colordist
from compareDataModel import _predict_rdist_plate
def fakeDensData(parser):
    (options,args)= parser.parse_args()
    if len(args) == 0:
        parser.print_help()
        return
    #Set up density model
    if options.model.lower() == 'hwr':
        densfunc= _HWRDensity
        params= numpy.log(numpy.array([0.3,2.5,1.]))
    elif options.model.lower() == 'flare':
        densfunc= _FlareDensity
        params= numpy.log(numpy.array([0.3,2.5,2.5]))
    elif options.model.lower() == 'tiedflare':
        densfunc= _TiedFlareDensity
        params= numpy.log(numpy.array([0.3,2.5]))
    elif options.model.lower() == 'twovertical':
        densfunc= _TwoVerticalDensity
        params= numpy.log(numpy.array([0.3,0.8,2.5,1.05]))
    #Load selection function
    sf= segueSelect.segueSelect(type_bright=options.sel_bright,
                                type_faint=options.sel_faint,
                                sample=options.sample)
    platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                     degree=True)
    if options.sample.lower() == 'g':
        rmin, rmax= 14.5, 20.2
        grmin, grmax= 0.48, 0.55
    elif options.sample.lower() == 'k':
        rmin, rmax= 14.5, 19.
        grmin, grmax= 0.55, 0.75
    feh= -0.15
    colordist= _const_colordist
    #Calculate the r-distribution for each plate
    nrs= 1001
    ngr= 11
    rs= numpy.linspace(rmin,rmax,nrs)
    rdists= numpy.zeros((len(sf.plates),nrs,ngr))
    for ii in range(len(sf.plates)):
        rdists[ii,:,:]= _predict_rdist_plate(rs,densfunc,params,rmin,rmax,
                                             platelb[ii,0],platelb[ii,1],
                                             grmin,grmax,
                                             feh,colordist,sf,sf.plates[ii],
                                             dontmarginalizecolor=True)
        if 'faint' in sf.platestr[ii].programname:
            rdists[ii,(rs < 17.8),:]= 0.
        elif not 'faint' in sf.platestr[ii].programname:
            rdists[ii,(rs > 17.8),:]= 0.
    numbers= numpy.sum(rdists,axis=2)
    numbers= numpy.sum(numbers,axis=1)
    numbers= numpy.cumsum(numbers)
    numbers/= numbers[-1]
    rdists= numpy.cumsum(rdists,axis=1)
    for ii in range(len(sf.plates)):
        rdists[ii,:,:]/= rdists[ii,-1,:]
    #Now sample until we're done
    out= []
    while len(out) < options.nsamples:
        #First sample a plate
        ran= numpy.random.uniform()
        kk= 0
        while numbers[kk] < ran: kk+= 1
        #plate==kk, now sample from the rdist of this plate
        #first sample a color, assuming constant
        gr=  int(numpy.floor(numpy.random.uniform()*ngr))
        ran= numpy.random.uniform()
        jj= 0
        while rdists[kk,jj,gr] < ran: jj+= 1
        #r=jj
        out.append([rs[jj],platelb[kk,0],platelb[kk,1]])
    #Save as pickle
    savefile= open(args[0],'wb')
    pickle.dump(out,savefile)
    savefile.close()

def get_options():
    usage = "usage: %prog [options] <savefilename>\n\nsavefilename= name of the file that the fake data will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("--model",dest='model',default='HWR',
                      help="Model to fit")
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--metal",dest='metal',default='rich',
                      help="Use metal-poor or rich sample ('poor', 'rich' or 'all')")
    parser.add_option("--sel_bright",dest='sel_bright',default='constant',
                      help="Selection function to use ('constant', 'r', 'platesn_r')")
    parser.add_option("--sel_faint",dest='sel_faint',default='r',
                      help="Selection function to use ('constant', 'r', 'platesn_r')")
    parser.add_option("-n","--nsamples",dest='nsamples',type='int',
                      default=5000,
                      help="Number of fake data points to return")
    return parser

if __name__ == '__main__':
    fakeDensData(get_options())
