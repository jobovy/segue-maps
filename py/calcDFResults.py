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
def resultsToInit(options,args,boot=True):    
    #First calcDFResults
    out= calcDFResults(options,args,boot=boot)
    #Then store
    sol= []
    for ii in range(len(out['feh'])):
        sol.extend([numpy.log(out['hr'][ii]/_REFR0),
                    numpy.log(out['sr'][ii]/_REFV0),
                    numpy.log(out['sz'][ii]/_REFV0),
                    numpy.log(out['hsr'][ii]/_REFR0),
                    numpy.log(out['hsz'][ii]/_REFR0),
                    out['outfrac'][ii]])
    sol.extend([numpy.log(out['rd_m']),
                out['vc_m']/_REFV0,
                numpy.log(out['zh_m']),
                out['dlnvcdlnr_m']*30.,
                out['plhalo_m']])
    #Save
    save_pickles(options.init,sol)
    return None                      
def calcDFResults(options,args,boot=True,nomedian=False):
    if len(args) == 2 and options.sample == 'gk':
        toptions= copy.copy(options)
        toptions.sample= 'g'
        toptions.select= 'all'
        outg= calcDFResults(toptions,[args[0]],boot=boot,nomedian=True)
        toptions.sample= 'k'
        toptions.select= 'program'
        outk= calcDFResults(toptions,[args[1]],boot=boot,nomedian=True)
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
    chi2s= []
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
        else:
            sols.append(pickle.load(savefile))
            chi2s.append(pickle.load(savefile))
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
    zmedians= []
    #Basic parameters
    hrs= []
    srs= []
    szs= []
    hsrs= []
    hszs= []
    outfracs= []
    rds= []
    rdexps= []
    vcs= []
    zhs= []
    zhexps= []
    dlnvcdlnrs= []
    plhalos= []
    #derived parameters
    surfzs= []
    surfz800s= []
    surfzdisks= []
    massdisks= []
    rhoos= []
    rhooalts= []
    rhodms= []
    vcdvcros= []
    vcdvcs= []
    mloglikemins= []
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
            zmedians.append(numpy.median(numpy.fabs(data.zc+_ZSUN)))
            #vc
            s= get_potparams(sols[solindx],options,1)
            ro= get_ro(sols[solindx],options)
            if options.fixvo:
                vcs.append(options.fixvo*_REFV0)
            else:
                vcs.append(s[1]*_REFV0)
            #rd
            rds.append(numpy.exp(s[0]))
            #zh
            zhs.append(numpy.exp(s[2-(1-(options.fixvo is None))]))
            #rdexp & zhexp
            if 'mpdisk' in options.potential.lower() or 'mwpotential' in options.potential.lower():
                mp= potential.MiyamotoNagaiPotential(a=rds[-1],b=zhs[-1])
                #rdexp
                f= mp.dens(1.,0.125)
                dr= 10.**-3.
                df= (mp.dens(1.+dr/2.,0.125)-mp.dens(1.-dr/2.,0.125))/dr
                rdexps.append(-f/df)
                #zhexp
                if options.sample == 'g': tz= 1.1/_REFR0/ro
                elif options.sample == 'k': tz= 0.84/_REFR0/ro
                f= mp.dens(1.,tz)
                dz= 10.**-3.
                df= (mp.dens(1.,tz+dz/2.)-mp.dens(1.,tz-dz/2.))/dz
                zhexps.append(-f/df)
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
            #surfz800
            surfz800s.append(2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot)),0.,0.8/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro)
            #surzdisk
            if 'mpdisk' in options.potential.lower() or 'mwpotential' in options.potential.lower():
                surfzdisks.append(2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot[0])),0.,options.height/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro)
                surfzdiskzm= 2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot[0])),0.,tz)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro
            #rhooalt
            if options.potential.lower() == 'mpdiskplhalofixbulgeflat':
                rhooalts.append(rhoos[-1]-pot[0].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.+surfzdiskzm/2./zhexps[-1]/ro/_REFR0/1000./(1.-numpy.exp(-tz/zhexps[-1])))
            #massdisk
            rhod= surfzdiskzm/2./zhexps[-1]/ro/_REFR0/1000./(1.-numpy.exp(-tz/zhexps[-1]))
            massdisks.append(rhod*2.*zhexps[-1]*numpy.exp(1./rdexps[-1])*rdexps[-1]**2.*2.*numpy.pi*(ro*_REFR0)**3./10.)
            #plhalo
            if options.potential.lower() == 'mpdiskplhalofixbulgeflat':
                plhalos.append(pot[1].alpha)
            #dlnvcdlnr
            dlnvcdlnrs.append(potential.dvcircdR(pot,1.))
            #vcdvc
            vcdvcros.append(pot[0].vcirc(1.)/potential.vcirc(pot,1.))
            vcdvcs.append(pot[0].vcirc(2.2*rdexps[-1])/potential.vcirc(pot,2.2*rdexps[-1]))
            mloglikemins.append(chi2s[solindx])
    #Gather
    fehs= numpy.array(fehs)
    afes= numpy.array(afes)
    zmedians= numpy.array(zmedians)
    ndatas= numpy.array(ndatas)
    #Basic parameters
    hrs= numpy.array(hrs)
    srs= numpy.array(srs)
    szs= numpy.array(szs)
    hsrs= numpy.array(hsrs)
    hszs= numpy.array(hszs)
    outfracs= numpy.array(outfracs)
    vcs= numpy.array(vcs)
    rds= numpy.array(rds)
    zhs= numpy.array(zhs)
    rdexps= numpy.array(rdexps)
    zhexps= numpy.array(zhexps)
    dlnvcdlnrs= numpy.array(dlnvcdlnrs)
    plhalos= numpy.array(plhalos)
    #derived parameters
    surfzs= numpy.array(surfzs)
    surfz800s= numpy.array(surfz800s)
    surfzdisks= numpy.array(surfzdisks)
    massdisks= numpy.array(massdisks)
    rhoos= numpy.array(rhoos)
    rhooalts= numpy.array(rhooalts)
    rhodms= numpy.array(rhodms)
    vcdvcros= numpy.array(vcdvcros)
    vcdvcs= numpy.array(vcdvcs)
    rexps= numpy.sqrt(2.)*(rds+zhs)/2.2
    mloglikemins= numpy.array(mloglikemins)
    #Load into dictionary
    out= {}
    out['feh']= fehs
    out['afe']= afes
    out['zmedian']= zmedians
    out['ndata']= ndatas
    out['hr']= hrs
    out['sr']= srs
    out['sz']= szs
    out['hsr']= hsrs
    out['hsz']= hszs
    out['outfrac']= outfracs
    out['vc']= vcs
    out['rd']= rds
    out['zh']= zhs
    out['rdexp']= rdexps
    out['zhexp']= zhexps
    out['dlnvcdlnr']= dlnvcdlnrs
    out['plhalo']= plhalos
    out['surfz']= surfzs
    out['surfz800']= surfz800s
    out['surfzdisk']= surfzdisks
    out['massdisk']= massdisks
    out['rhoo']= rhoos
    out['rhooalt']= rhooalts
    out['rhodm']= rhodms
    out['vcdvc']= vcdvcs
    out['vcdvcro']= vcdvcros
    out['rexp']= rexps
    out['mloglikemin']= mloglikemins
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
