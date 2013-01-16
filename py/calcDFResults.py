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
def setup_options(options):
    if options is None:
        parser= get_options()
        options, args= parser.parse_args([])
    #Set up to default fit
    options.potential= 'mpdiskplhalofixbulgeflat'
    options.aAmethod= 'staeckelg'
    options.dfeh=0.1
    options.dafe=0.05
    options.ngl= 20
    options.singles= True
    return options
def calcDFResults(options,args,boot=True,nomedian=False):
    if len(args) == 2 and options.sample == 'gk':
        options.sample= 'g'
        options.select= 'all'
        outg= calcDFResults(options,[args[0]],boot=boot,nomedian=True)
        options.sample= 'k'
        options.select= 'program'
        outk= calcDFResults(options,[args[1]],boot=boot,nomedian=True)
        #Combine
        out= {}
        for k in outg.keys():
            valg= outg[k]
            valk= outk[k]
            val= numpy.zeros(len(valg)+len(valk))
            val[0:len(valg)]= valg
            val[len(valg):len(valg)+len(valk)]= valk
            out[k]= val
        if nomedian: return out
        else: return add_median(out,boot=boot)
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
        else:
            sols.append(pickle.load(savefile))
            savefile.close()
        #Load samples as well
        if options.mcsample:
            #Do the same for init
            spl= initname.split('.')
            newname= ''
            for jj in range(len(spl)-1):
                newname+= spl[jj]
                if not jj == len(spl)-2: newname+= '.'
            newname+= '_%i.' % ii
            newname+= spl[-1]
            options.init= newname
    mapfehs= monoAbundanceMW.fehs()
    mapafes= monoAbundanceMW.afes()
    #Now plot
    #Run through the pixels and gather
    fehs= []
    afes= []
    ndatas= []
    #Basic parameters
    hrs= []
    srs= []
    szs= []
    hsrs= []
    hszs= []
    outfracs= []
    rds= []
    vcs= []
    zhs= []
    dlnvcdlnrs= []
    plhalos= []
    #derived parameters
    surfzs= []
    surfzdisks= []
    rhoos= []
    rhodms= []
    vcdvcros= []
    vcdvcs= []
    for ii in range(tightbinned.npixfeh()):
        for jj in range(tightbinned.npixafe()):
            data= binned(tightbinned.feh(ii),tightbinned.afe(jj))
            if len(data) < options.minndata:
                continue
            #Find abundance indx
            fehindx= binned.fehindx(tightbinned.feh(ii))#Map onto regular binning
            afeindx= binned.afeindx(tightbinned.afe(jj))
            solindx= abindx[fehindx,afeindx]
            monoabindx= numpy.argmin((tightbinned.feh(ii)-mapfehs)**2./0.01 \
                                         +(tightbinned.afe(jj)-mapafes)**2./0.0025)
            if sols[solindx] is None:
                continue
            fehs.append(tightbinned.feh(ii))
            afes.append(tightbinned.afe(jj))
            #vc
            s= get_potparams(sols[solindx],options,1)
            if options.fixvo:
                vcs.append(options.fixvo*_REFV0)
            else:
                vcs.append(s[1]*_REFV0)
            #rd
            rds.append(numpy.exp(s[0]))
            #zh
            zhs.append(numpy.exp(s[2-(1-(options.fixvo is None))]))
            #ndata
            ndatas.append(len(data))
            #hr
            dfparams= get_dfparams(sols[solindx],0,options)
            if options.relative:
                thishr= monoAbundanceMW.hr(mapfehs[monoabindx],mapafes[monoabindx])
                hrs.append(dfparams[0]*_REFR0/thishr)
            else:
                hrs.append(dfparams[0]*_REFR0)
            #sz
            if options.relative:
                thissz= monoAbundanceMW.sigmaz(mapfehs[monoabindx],mapafes[monoabindx])
                szs.append(dfparams[2]*_REFV0/thissz)
            else:
                szs.append(dfparams[2]*_REFV0)
            #sr
            if options.relative:
                thissr= monoAbundanceMW.sigmaz(mapfehs[monoabindx],mapafes[monoabindx])*2.#BOVY: UPDATE
                srs.append(dfparams[1]*_REFV0/thissr)
            else:
                srs.append(dfparams[1]*_REFV0)
            #hsr
            hsrs.append(dfparams[3]*_REFR0)
            #hsz
            hszs.append(dfparams[4]*_REFR0)
            #outfrac
            outfracs.append(dfparams[5])
            #rhodm
            #Setup potential
            pot= setup_potential(sols[solindx],options,1)
            vo= get_vo(sols[solindx],options,1)
            ro= get_ro(sols[solindx],options)
            if 'mwpotential' in options.potential.lower():
                rhodms.append(pot[1].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.)
            elif options.potential.lower() == 'mpdiskplhalofixbulgeflat':
                rhodms.append(pot[1].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.)
            elif options.potential.lower() == 'mpdiskflplhalofixplfixbulgeflat':
                rhodms.append(pot[1].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.)
            #rhoo
            rhoos.append(potential.evaluateDensities(1.,0.,pot)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.)
            #surfz
            surfzs.append(2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot)),0.,options.height/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro)
            #surzdisk
            if 'mpdisk' in options.potential.lower() or 'mwpotential' in options.potential.lower():
                surfzdisks.append(2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot[0])),0.,options.height/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro)
            #plhalo
            if options.potential.lower() == 'mpdiskplhalofixbulgeflat':
                plhalos.append(pot[1].alpha)
            #dlnvcdlnr
            dlnvcdlnrs.append(potential.dvcircdR(pot,1.))
            #vcdvc
            vcdvcros.append(pot[0].vcirc(1.)/potential.vcirc(pot,1.))
            vcdvcs.append(pot[0].vcirc(numpy.sqrt(2.)*(rds[-1]+zhs[-1]))/potential.vcirc(pot,numpy.sqrt(2.)*(rds[-1]+zhs[-1])))
    #Gather
    fehs= numpy.array(fehs)
    afes= numpy.array(afes)
    ndatas= numpy.array(ndatas)
    #Basic parameters
    hrs= numpy.array(hrs)
    srs= numpy.array(srs)
    szs= numpy.array(szs)
    hsrs= numpy.array(hsrs)
    hszs= numpy.array(hszs)
    outfracs= numpy.array(outfracs)
    rds= numpy.array(rds)
    vcs= numpy.array(vcs)
    zhs= numpy.array(zhs)
    dlnvcdlnrs= numpy.array(dlnvcdlnrs)
    plhalos= numpy.array(plhalos)
    #derived parameters
    surfzs= numpy.array(surfzs)
    surfzdisks= numpy.array(surfzdisks)
    rhoos= numpy.array(rhoos)
    rhodms= numpy.array(rhodms)
    vcdvcros= numpy.array(vcdvcros)
    vcdvcs= numpy.array(vcdvcs)
    rexps= numpy.sqrt(2.)*(rds+zhs)/2.2
    #Load into dictionary
    out= {}
    out['feh']= fehs
    out['afe']= afes
    out['ndata']= ndatas
    out['hr']= hrs
    out['sr']= srs
    out['sz']= szs
    out['hsr']= hsrs
    out['hsz']= hszs
    out['outfrac']= outfracs
    out['rd']= rds
    out['vc']= vcs
    out['zh']= zhs
    out['dlnvcdlnr']= dlnvcdlnrs
    out['plhalo']= plhalos
    out['surfz']= surfzs
    out['surfzdisk']= surfzdisks
    out['rhoo']= rhoos
    out['rhodm']= rhodms
    out['vcdvc']= vcdvcs
    out['vcdvcro']= vcdvcros
    out['rexp']= rexps
    if nomedian: return out
    else: return add_median(out,boot=boot)

def add_median(out,boot=False):
    for key in out.keys():
        data= out[key]
        out[key+'_m']= numpy.median(out[key])
        if boot:
            out[key+'_merr']= booterr(out[key],numpy.median)
        else:
            out[key+'_merr']= jackerr(out[key],numpy.median)
    return out

def jackerr(data,func):
    ests= numpy.zeros_like(data)
    for ii in range(len(data)):
        tdata= list(copy.copy(data))
        dummy= tdata.pop(ii)
        ests[ii]= func(tdata)
    return numpy.sqrt((len(data)-1.)*numpy.var(ests))

def booterr(data,func,nboot=1000):
    ests= numpy.zeros(nboot)
    for ii in range(nboot):
        #Resample
        tdata= numpy.zeros_like(data)
        for jj in range(len(data)):
            tdata[jj]= data[int(numpy.floor(numpy.random.uniform()*len(data)))]
        ests[ii]= func(tdata)
    return numpy.sqrt(numpy.var(ests,ddof=1.))
