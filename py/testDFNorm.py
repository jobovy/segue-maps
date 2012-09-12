#e.g., python testDFNorm.py --dfeh=0.5 --dafe=0.25 --singlefeh=-0.2 --singleafe=0. -f testFakeDFData.fits --mcall --mcout --savenorm=testFakeNorm100k.sav -o ~/Desktop/test.png -t vo
import os, os.path
import sys
import copy
import tempfile
import math
import numpy
from scipy import optimize, interpolate, linalg
from scipy.maxentropy import logsumexp
import cPickle as pickle
from optparse import OptionParser
import multi
import multiprocessing
from galpy.util import bovy_coords, bovy_plot, save_pickles
from matplotlib import pyplot
from matplotlib.ticker import NullFormatter
from galpy import potential
from galpy.actionAngle_src.actionAngleAdiabaticGrid import  actionAngleAdiabaticGrid
from galpy.df_src.quasiisothermaldf import quasiisothermaldf
import bovy_mcmc
import acor
from galpy.util import save_pickles
import monoAbundanceMW
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE, \
    segueSelect, _mr_gi, _gi_gr, _ERASESTR, _append_field_recarray
from fitDensz import cb, _ZSUN, DistSpline, _ivezic_dist, _NDS
from compareDataModel import _predict_rdist_plate
from pixelFitDens import pixelAfeFeh
from pixelFitDF import *
def testDFNorm(options,args):
    #Read the data
    print "Reading the data ..."
    if options.sample.lower() == 'g':
        if not options.fakedata is None:
            raw= read_gdwarfs(options.fakedata,logg=True,ebv=True,sn=options.snmin,nosolar=True)
        elif options.select.lower() == 'program':
            raw= read_gdwarfs(_GDWARFFILE,logg=True,ebv=True,sn=options.snmin,nosolar=True)
        else:
            raw= read_gdwarfs(logg=True,ebv=True,sn=options.snmin,nosolar=True)
    elif options.sample.lower() == 'k':
        if options.select.lower() == 'program':
            raw= read_kdwarfs(_KDWARFFILE,logg=True,ebv=True,sn=options.snmin,nosolar=True)
        else:
            raw= read_kdwarfs(logg=True,ebv=True,sn=options.snmin,nosolar=True)
    if not options.bmin is None:
        #Cut on |b|
        raw= raw[(numpy.fabs(raw.b) > options.bmin)]
    if not options.fehmin is None:
        raw= raw[(raw.feh >= options.fehmin)]
    if not options.fehmax is None:
        raw= raw[(raw.feh < options.fehmax)]
    if not options.afemin is None:
        raw= raw[(raw.afe >= options.afemin)]
    if not options.afemax is None:
        raw= raw[(raw.afe < options.afemax)]
    if not options.plate is None and not options.loo:
        raw= raw[(raw.plate == options.plate)]
    elif not options.plate is None:
        raw= raw[(raw.plate != options.plate)]
    #Bin the data
    binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
    #Map the bins with ndata > minndata in 1D
    fehs, afes= [], []
    for ii in range(len(binned.fehedges)-1):
        for jj in range(len(binned.afeedges)-1):
            data= binned(binned.feh(ii),binned.afe(jj))
            if len(data) < options.minndata:
                continue
            fehs.append(binned.feh(ii))
            afes.append(binned.afe(jj))
    nabundancebins= len(fehs)
    fehs= numpy.array(fehs)
    afes= numpy.array(afes)
    if not options.singlefeh is None:
        if options.loo:
            pass
        else:
            #Set up single feh
            indx= binned.callIndx(options.singlefeh,options.singleafe)
            if numpy.sum(indx) == 0:
                raise IOError("Bin corresponding to singlefeh and singleafe is empty ...")
            data= copy.copy(binned.data[indx])
            print "Using %i data points ..." % (len(data))
            #Bin again
            binned= pixelAfeFeh(data,dfeh=options.dfeh,dafe=options.dafe)
            fehs, afes= [], []
            for ii in range(len(binned.fehedges)-1):
                for jj in range(len(binned.afeedges)-1):
                    data= binned(binned.feh(ii),binned.afe(jj))
                    if len(data) < options.minndata:
                        continue
                    fehs.append(binned.feh(ii))
                    afes.append(binned.afe(jj))
            nabundancebins= len(fehs)
            fehs= numpy.array(fehs)
            afes= numpy.array(afes)
    #Setup everything for the selection function
    print "Setting up stuff for the normalization integral ..."
    normintstuff= setup_normintstuff(options,raw,binned,fehs,afes)
    if not options.init is None:
        #Load initial parameters from file
        savefile= open(options.init,'rb')
        params= pickle.load(savefile)
        savefile.close()
    else:
        #First initialization
        params= initialize(options,fehs,afes)
    #Now perform tests
    if options.type.lower() == 'hr':
        testDFNormhr(params,fehs,afes,binned,options,normintstuff)
    elif options.type.lower() == 'sr':
        testDFNormsr(params,fehs,afes,binned,options,normintstuff)
    elif options.type.lower() == 'vo':
        testDFNormvo(params,fehs,afes,binned,options,normintstuff)

def testDFNormhr(params,fehs,afes,binned,options,normintstuff):
    """Test how the normalization fares wrt hr changes"""
    #setup potential
    pot= setup_potential(params,options,len(fehs))
    aA= setup_aA(pot,options)
    ro= get_ro(params,options)
    vo= get_vo(params,options,len(fehs))  
    #Assume there is only a single bin
    indx= 0
    dfparams= get_dfparams(params,indx,options,log=False)
    defhr= dfparams[0]
    nalt= 6
    hrs= numpy.linspace(0.5,1.5,nalt)
    ns= [1000,10000,100000]
    #First calculate default
    if options.dfmodel.lower() == 'qdf':
        #Normalize
        hr= dfparams[0]/ro
        sr= dfparams[1]/vo
        sz= dfparams[2]/vo
        hsr= dfparams[3]/ro
        hsz= dfparams[4]/ro
        #Setup
        qdf= quasiisothermaldf(hr,sr,sz,hsr,hsz,pot=pot,aA=aA,cutcounter=True)
    defNorm= numpy.zeros((nalt,len(ns)))+numpy.random.random(size=(nalt,len(ns)))
    for ii, n in enumerate(ns):
        print ii, n
        options.nmc= n
        thisnormintstuff= copy.deepcopy(normintstuff)
        thisnormintstuff[indx].mock= normintstuff[indx].mock[0:n]
        defNorm[:,ii]= numpy.log(calc_normint(qdf,indx,thisnormintstuff,params,len(fehs),options))
    #Then calculate alternative models
    altNorm= numpy.zeros((nalt,len(ns)))+numpy.random.random(size=(nalt,len(ns)))
    for ii in range(nalt):
        if options.dfmodel.lower() == 'qdf':
            #Normalize
            hr= dfparams[0]/ro*hrs[ii]
            #Setup
            qdf= quasiisothermaldf(hr,sr,sz,hsr,hsz,pot=pot,aA=aA,
                                   cutcounter=True)
        for jj, n in enumerate(ns):
            if n > len(normintstuff[indx].mock):
                altNorm[ii,jj]= numpy.nan
                continue
            print ii, jj, n
            options.nmc= n
            thisnormintstuff= copy.deepcopy(normintstuff)
            thisnormintstuff[indx].mock= normintstuff[indx].mock[0:n+1]
            altNorm[ii,jj]= numpy.log(calc_normint(qdf,indx,thisnormintstuff,
                                                   params,len(fehs),options))
    #Plot
    left, bottom, width, height= 0.1, 0.3, 0.8, 0.6
    axTop= pyplot.axes([left,bottom,width,height])
    left, bottom, width, height= 0.1, 0.1, 0.8, 0.2
    axSign= pyplot.axes([left,bottom,width,height])
    fig= pyplot.gcf()
    fig.sca(axTop)
    pyplot.ylabel(r'$|\Delta \chi^2|$')
    pyplot.xlim(ns[0]/5.,ns[-1]*5.)
    nullfmt   = NullFormatter()         # no labels
    axTop.xaxis.set_major_formatter(nullfmt)
    pyplot.loglog(numpy.tile(numpy.array(ns),(nalt,1)).T,
                  numpy.fabs((defNorm-altNorm)*10000.).T,
                  marker='o',linestyle='none')
    fig.sca(axSign)
    pyplot.semilogx(numpy.tile(numpy.array(ns),(nalt,1)).T\
                        *(1.+0.3*(numpy.random.uniform(size=(len(ns),nalt))-0.5)),
                    numpy.fabs((defNorm-altNorm)).T/(defNorm-altNorm).T,
                    marker='o',linestyle='none')
    pyplot.xlim(ns[0]/5.,ns[-1]*5.)
    pyplot.ylim(-1.99,1.99)
    pyplot.xlabel(r'$N$')
    pyplot.ylabel(r'$\mathrm{sgn}(\Delta \chi^2)$')
    bovy_plot.bovy_end_print(options.outfilename)
            
def testDFNormsr(params,fehs,afes,binned,options,normintstuff):
    """Test how the normalization fares wrt sr changes"""
    #setup potential
    pot= setup_potential(params,options,len(fehs))
    aA= setup_aA(pot,options)
    ro= get_ro(params,options)
    vo= get_vo(params,options,len(fehs))  
    #Assume there is only a single bin
    indx= 0
    dfparams= get_dfparams(params,indx,options,log=False)
    defhr= dfparams[0]
    nalt= 6
    srs= numpy.linspace(0.5,1.5,nalt)
    ns= [1000,10000,100000]
    #First calculate default
    if options.dfmodel.lower() == 'qdf':
        #Normalize
        hr= dfparams[0]/ro
        sr= dfparams[1]/vo
        sz= dfparams[2]/vo
        hsr= dfparams[3]/ro
        hsz= dfparams[4]/ro
        #Setup
        qdf= quasiisothermaldf(hr,sr,sz,hsr,hsz,pot=pot,aA=aA,cutcounter=True)
    defNorm= numpy.zeros((nalt,len(ns)))+numpy.random.random(size=(nalt,len(ns)))
    for ii, n in enumerate(ns):
        print ii, n
        options.nmc= n
        thisnormintstuff= copy.deepcopy(normintstuff)
        thisnormintstuff[indx].mock= normintstuff[indx].mock[0:n]
        defNorm[:,ii]= numpy.log(calc_normint(qdf,indx,thisnormintstuff,params,len(fehs),options))
    #Then calculate alternative models
    altNorm= numpy.zeros((nalt,len(ns)))+numpy.random.random(size=(nalt,len(ns)))
    for ii in range(nalt):
        if options.dfmodel.lower() == 'qdf':
            #Normalize
            sr= dfparams[1]/vo*srs[ii]
            #Setup
            qdf= quasiisothermaldf(hr,sr,sz,hsr,hsz,pot=pot,aA=aA,
                                   cutcounter=True)
        for jj, n in enumerate(ns):
            if n > len(normintstuff[indx].mock):
                altNorm[ii,jj]= numpy.nan
                continue
            print ii, jj, n
            options.nmc= n
            thisnormintstuff= copy.deepcopy(normintstuff)
            thisnormintstuff[indx].mock= normintstuff[indx].mock[0:n+1]
            altNorm[ii,jj]= numpy.log(calc_normint(qdf,indx,thisnormintstuff,
                                                   params,len(fehs),options))
    #Plot
    left, bottom, width, height= 0.1, 0.3, 0.8, 0.6
    axTop= pyplot.axes([left,bottom,width,height])
    left, bottom, width, height= 0.1, 0.1, 0.8, 0.2
    axSign= pyplot.axes([left,bottom,width,height])
    fig= pyplot.gcf()
    fig.sca(axTop)
    pyplot.ylabel(r'$|\Delta \chi^2|$')
    pyplot.xlim(ns[0]/5.,ns[-1]*5.)
    nullfmt   = NullFormatter()         # no labels
    axTop.xaxis.set_major_formatter(nullfmt)
    pyplot.loglog(numpy.tile(numpy.array(ns),(nalt,1)).T,
                  numpy.fabs((defNorm-altNorm)*10000.).T,
                  marker='o',linestyle='none')
    fig.sca(axSign)
    pyplot.semilogx(numpy.tile(numpy.array(ns),(nalt,1)).T\
                        *(1.+0.4*(numpy.random.uniform(size=(len(ns),nalt))-0.5)),
                    numpy.fabs((defNorm-altNorm)).T/(defNorm-altNorm).T,
                    marker='o',linestyle='none')
    pyplot.xlim(ns[0]/5.,ns[-1]*5.)
    pyplot.ylim(-1.99,1.99)
    pyplot.xlabel(r'$N$')
    pyplot.ylabel(r'$\mathrm{sgn}(\Delta \chi^2)$')
    bovy_plot.bovy_end_print(options.outfilename)
            
def testDFNormvo(params,fehs,afes,binned,options,normintstuff):
    """Test how the normalization fares wrt hr changes"""
    #setup potential
    pot= setup_potential(params,options,len(fehs))
    aA= setup_aA(pot,options)
    ro= get_ro(params,options)
    vo= get_vo(params,options,len(fehs))
    #Assume there is only a single bin
    indx= 0
    dfparams= get_dfparams(params,indx,options,log=False)
    nalt= 6
    vos= numpy.linspace(0.5,1.5,nalt)
    ns= [1000,10000,100000]#,1000000]
    #First calculate default
    if options.dfmodel.lower() == 'qdf':
        #Normalize
        hr= dfparams[0]/ro
        sr= dfparams[1]/vo
        sz= dfparams[2]/vo
        hsr= dfparams[3]/ro
        hsz= dfparams[4]/ro
        #Setup
        qdf= quasiisothermaldf(hr,sr,sz,hsr,hsz,pot=pot,aA=aA,cutcounter=True)
    defNorm= numpy.zeros((nalt,len(ns)))
    for ii, n in enumerate(ns):
        print ii, n
        options.nmc= n
        thisnormintstuff= copy.deepcopy(normintstuff)
        thisnormintstuff[indx].mock= normintstuff[indx].mock[0:n+1]
        defNorm[:,ii]= numpy.log(calc_normint(qdf,indx,thisnormintstuff,params,len(fehs),options))
    print defNorm[0,:]
    #Then calculate alternative models
    altNorm= numpy.zeros((nalt,len(ns)))
    for ii in range(nalt):
        potparams= list(get_potparams(params,options,len(fehs)))
        potparams[0]= vo*vos[ii]
        params= set_potparams(potparams,params,options,len(fehs))
        if options.dfmodel.lower() == 'qdf':
            #Normalize
            sr= dfparams[1]/vo/vos[ii]
            sz= dfparams[2]/vo/vos[ii]
            #Setup
            qdf= quasiisothermaldf(hr,sr,sz,hsr,hsz,pot=pot,aA=aA,
                                   cutcounter=True)
        for jj, n in enumerate(ns):
            if n > len(normintstuff[indx].mock):
                altNorm[ii,jj]= numpy.nan
                continue
            print ii, jj, n
            options.nmc= n
            thisnormintstuff= copy.deepcopy(normintstuff)
            thisnormintstuff[indx].mock= normintstuff[indx].mock[0:n]
            altNorm[ii,jj]= numpy.log(calc_normint(qdf,indx,thisnormintstuff,
                                                   params,len(fehs),options))
    #Plot
    left, bottom, width, height= 0.1, 0.3, 0.8, 0.6
    axTop= pyplot.axes([left,bottom,width,height])
    left, bottom, width, height= 0.1, 0.1, 0.8, 0.2
    axSign= pyplot.axes([left,bottom,width,height])
    fig= pyplot.gcf()
    fig.sca(axTop)
    pyplot.ylabel(r'$|\Delta \chi^2|$')
    pyplot.xlim(ns[0]/5.,ns[-1]*5.)
    nullfmt   = NullFormatter()         # no labels
    axTop.xaxis.set_major_formatter(nullfmt)
    pyplot.loglog(numpy.tile(numpy.array(ns),(nalt,1)).T,
                  numpy.fabs((defNorm-altNorm)*10000.).T,
                  marker='o',linestyle='none')
    fig.sca(axSign)
    pyplot.semilogx(numpy.tile(numpy.array(ns),(nalt,1)).T\
                        *(1.+0.4*(numpy.random.uniform(size=(len(ns),nalt))-0.5)),
                    numpy.fabs((defNorm-altNorm)).T/(defNorm-altNorm).T,
                    marker='o',linestyle='none')
    pyplot.xlim(ns[0]/5.,ns[-1]*5.)
    pyplot.ylim(-1.99,1.99)
    pyplot.xlabel(r'$N$')
    pyplot.ylabel(r'$\mathrm{sgn}(\Delta \chi^2)$')
    bovy_plot.bovy_end_print(options.outfilename)
            
if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    numpy.random.seed(options.seed)
    testDFNorm(options,args)

