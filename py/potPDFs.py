import os, os.path
import math
import numpy
from scipy import integrate
import cPickle as pickle
import monoAbundanceMW
from galpy.util import bovy_plot
from galpy import potential
from galpy.df import quasiisothermaldf
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE, \
    segueSelect, _mr_gi, _gi_gr, _ERASESTR, _append_field_recarray, \
    ivezic_dist_gr
from fitDensz import cb, _ZSUN, DistSpline, _ivezic_dist, _NDS
from pixelFitDens import pixelAfeFeh
from pixelFitDF import _REFV0, get_options, read_rawdata, get_potparams, \
    get_dfparams, _REFR0, get_vo, get_ro, setup_potential, setup_aA, \
    get_dvt
class potPDFs:
    """Class for representing potential PDFs"""
    def __init__(self,options,args,basic=True):
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
        vcs= []
        vc_samples= []
        rds= []
        rd_samples= []
        zhs= []
        zh_samples= []
        dlnvcdlnrs= []
        dlnvcdlnr_samples= []
        plhalos= []
        plhalo_samples= []
        dvts= []
        dvt_samples= []
        #derived
        surfzs= []
        surfz_samples= []
        surfz800s= []
        surfz800_samples= []
        surfzdisks= []
        surfzdisk_samples= []
        massdisks= []
        massdisk_samples= []
        rhodms= []
        rhodm_samples= []
        rhoos= []
        rhoo_samples= []
        rorss= []
        rors_samples= []
        vcdvcs= []
        vcdvc_samples= []
        vcdvcros= []
        vcdvcro_samples= []
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
                print len(rds)
                try:
                    pot= setup_potential(sols[solindx],options,1,
                                         interpDens=True,
                                         interpdvcircdr=True,returnrawpot=True)
                except RuntimeError:
                    print "A bin has an unphysical potential ..."
                    continue
                fehs.append(tightbinned.feh(ii))
                afes.append(tightbinned.afe(jj))
                ndatas.append(len(data))
                #vc
                s= get_potparams(sols[solindx],options,1)
                ro= get_ro(sols[solindx],options)
                vo= get_vo(sols[solindx],options,1)
                if options.fixvo:
                    vcs.append(options.fixvo*_REFV0)
                else:
                    vcs.append(s[1]*_REFV0)
                #rd
                rds.append(numpy.exp(s[0])*_REFR0*ro)
                #zh
                zhs.append(numpy.exp(s[2])*_REFR0*ro)
                #dlnvcdlnrs
                dlnvcdlnrs.append(s[3]/30.)
                #plhalos
                plhalos.append(s[4])
                rorss.append((1.-plhalos[-1])/(plhalos[-1]-3.))
                #dvt
                if options.fitdvt:
                    dvts.append(get_dvt(sols[solindx],options))
                #rhodm
                if options.potential.lower() == 'dpdiskplhalofixbulgeflat' \
                    or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgas' \
                    or options.potential.lower() == 'dpdiskflplhalofixbulgeflatwgas':
                    rhodms.append(pot[1].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.)
                #rhoo
                rhoos.append(potential.evaluateDensities(1.,0.,pot)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.)
                #surfz
                surfzs.append(2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot)),0.,options.height/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro)
                #surfz800
                surfz800s.append(2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot)),0.,0.8/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro)
                #surfzdisk
                if 'dpdisk' in options.potential.lower():
                    surfzdisks.append(2.*pot[0].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*zhs[-1]*1000.)
                #massdisk
                if options.potential.lower() == 'dpdiskplhalofixbulgeflat' \
                        or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgas' \
                        or options.potential.lower() == 'dpdiskflplhalofixbulgeflatwgas':
                    rhod= pot[0].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.
                massdisks.append(rhod*2.*zhs[-1]*numpy.exp(ro*_REFR0/rds[-1])*rds[-1]**2.*2.*numpy.pi/10.)
                #vcdvc
                if options.potential.lower() == 'dpdiskplhalofixbulgeflat':
                    vcdvcros.append(pot[0].vcirc(1.)/potential.vcirc(pot,1.))
                    vcdvcs.append(pot[0].vcirc(2.2*rds[-1]/ro/_REFR0)/potential.vcirc(pot,2.2*rds[-1]/ro/_REFR0))
                elif options.potential.lower() == 'dpdiskplhalofixbulgeflatwgas' \
                        or options.potential.lower() == 'dpdiskflplhalofixbulgeflatwgas':
                    vcdvcros.append(potential.vcirc([pot[0],pot[3]],1.)/potential.vcirc(pot,1.))
                    vcdvcs.append(potential.vcirc([pot[0],pot[3]],2.2*rds[-1]/ro/_REFR0)/potential.vcirc(pot,2.2*rds[-1]/ro/_REFR0))
                #Gather samples
                thisvc_samples= []
                thisrd_samples= []
                thiszh_samples= []
                thisdlnvcdlnr_samples= []
                thisplhalo_samples= []
                thisdvt_samples= []
                #derived
                thissurfzdisk_samples= []
                thismassdisk_samples= []
                thissurfz_samples= []
                thissurfz800_samples= []
                thisrhodm_samples= []
                thisrhoo_samples= []
                thisvcdvc_samples= []
                thisvcdvcro_samples= []
                thisrors_samples= []
                for kk in range(len(samples[0])):
                    if not basic:
                        try:
                            pot= setup_potential(samples[solindx][kk],options,1,
                                                 interpDens=True,
                                                 interpdvcircdr=True,returnrawpot=True)
                        except RuntimeError:
                            continue
                    s= get_potparams(samples[solindx][kk],options,1)
                    ro= get_ro(samples[solindx][kk],options)
                    vo= get_vo(samples[solindx][kk],options,1)
                    if options.fixvo:
                        thisvc_samples.append(options.fixvo*_REFV0)
                    else:
                        thisvc_samples.append(s[1]*_REFV0)
                    thisrd_samples.append(numpy.exp(s[0])*_REFR0)
                    thiszh_samples.append(numpy.exp(s[2])*_REFR0)
                    #dlnvcdlnrs
                    thisdlnvcdlnr_samples.append(s[3]/30.)
                    #plhalos
                    thisplhalo_samples.append(s[4])
                    thisrors_samples.append((1.-thisplhalo_samples[-1])/(thisplhalo_samples[-1]-3.))
                    #dvt
                    if options.fitdvt:
                        thisdvt_samples.append(get_dvt(sols[solindx],options))
                    if not basic:
                        #rhodm
                        if options.potential.lower() == 'dpdiskplhalofixbulgeflat' \
                                or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgas' \
                                or options.potential.lower() == 'dpdiskflplhalofixbulgeflatwgas':
                            thisrhodm_samples.append(pot[1].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.)
                        #rhoo
                        thisrhoo_samples.append(potential.evaluateDensities(1.,0.,pot)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.)
                        #surfz
                        thissurfz_samples.append(2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot)),0.,options.height/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro)
                        #surfz800
                        thissurfz800_samples.append(2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot)),0.,0.8/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro)
                        #surfzdisk
                        if 'dpdisk' in options.potential.lower():
                            thissurfzdisk_samples.append(2.*pot[0].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*thiszh_samples[-1]*1000.)
                        #massdisk
                        if options.potential.lower() == 'dpdiskplhalofixbulgeflat' \
                                or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgas' \
                                or options.potential.lower() == 'dpdiskflplhalofixbulgeflatwgas':
                            rhod= pot[0].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.
                        thismassdisk_samples.append(rhod*2.*thiszh_samples[-1]*numpy.exp(ro*_REFR0/thisrd_samples[-1])*thisrd_samples[-1]**2.*2.*numpy.pi/10.)
                        #vcdvc
                        if options.potential.lower() == 'dpdiskplhalofixbulgeflat':
                            thisvcdvcro_samples.append(pot[0].vcirc(1.)/potential.vcirc(pot,1.))
                            thisvcdvc_samples.append(pot[0].vcirc(2.2*rds[-1]/ro/_REFR0)/potential.vcirc(pot,2.2*rds[-1]/ro/_REFR0))
                        elif options.potential.lower() == 'dpdiskplhalofixbulgeflatwgas' \
                                or options.potential.lower() == 'dpdiskflplhalofixbulgeflatwgas':
                            thisvcdvcro_samples.append(potential.vcirc([pot[0],pot[3]],1.)/potential.vcirc(pot,1.))
                            thisvcdvc_samples.append(potential.vcirc([pot[0],pot[3]],2.2*rds[-1]/ro/_REFR0)/potential.vcirc(pot,2.2*rds[-1]/ro/_REFR0))
                vc_samples.append(thisvc_samples)
                rd_samples.append(thisrd_samples)
                zh_samples.append(thiszh_samples)
                dlnvcdlnr_samples.append(thisdlnvcdlnr_samples)
                plhalo_samples.append(thisplhalo_samples)
                rors_samples.append(thisrors_samples)
                dvt_samples.append(thisdvt_samples)
                #derived
                surfz_samples.append(thissurfz_samples)
                surfz800_samples.append(thissurfz800_samples)
                surfzdisk_samples.append(thissurfzdisk_samples)
                massdisk_samples.append(thismassdisk_samples)
                vcdvc_samples.append(vcdvc_samples)
                vcdvcro_samples.append(vcdvcro_samples)
                rhoo_samples.append(rhoo_samples)
                rhodm_samples.append(rhodm_samples)
        self._fehs= fehs
        self._afes= afes
        self._ndatas= ndatas
        self._vcs= vcs
        self._vc_samples= vc_samples
        self._rds= rds
        self._rd_samples= rd_samples
        self._zhs= zhs
        self._zh_samples= zh_samples
        self._dlnvcdlnrs= dlnvcdlnrs
        self._dlnvcdlnr_samples= dlnvcdlnr_samples
        self._plhalos= plhalos
        self._plhalo_samples= plhalo_samples
        self._surfzs= surfzs
        self._surfz_samples= surfz_samples
        self._surfz800s= surfz800s
        self._surfz800_samples= surfz800_samples
        self._surfzdisks= surfzdisks
        self._surfzdisk_samples= surfzdisk_samples
        self._massdisks= massdisks
        self._massdisk_samples= massdisk_samples
        self._rhoos= rhoos
        self._rhoo_samples= rhoo_samples
        self._rhodms= rhodms
        self._rhodm_samples= rhodm_samples
        self._vcdvcs= vcdvcs
        self._vcdvc_samples= vcdvc_samples
        self._vcdvcros= vcdvcros
        self._vcdvcro_samples= vcdvcro_samples
        self._rorss= rorss
        self._rors_samples= rors_samples
        self._dvts= dvts
        self._dvt_samples= dvt_samples
        return None

    def hist1d(self,param,*args,**kwargs):
        """param = string with parameter name"""
        overplot=False
        hists= []
        for ii in range(len(self._rds)):
            hists.append(bovy_plot.bovy_hist(self.__dict__['_%s_samples' % param][ii],
                                             *args,overplot=overplot,
                                             **kwargs))
            overplot= True
        tothist= [numpy.prod(numpy.array([h[0][ii] for h in hists],dtype='float')) for ii in range(len(hists[0][0]))]
        xs= (hists[0][1]+numpy.roll(hists[0][1],-1))[0:-1]/2.
        tothist/= numpy.sum(tothist)*(xs[1]-xs[0])
        bovy_plot.bovy_plot(xs,tothist,'k-',lw=2.,overplot=True)
        return (xs,tothist)

    def scatterplot(self,param1,param2,indx,*args,**kwargs):
        """param = string with parameter name, index=index of bin"""
        bovy_plot.scatterplot(numpy.array(self.__dict__['_%s_samples' % param1][indx]),
                              numpy.array(self.__dict__['_%s_samples' % param2][indx]),
                              *args,
                              **kwargs)
        bovy_plot.bovy_text(r'$[\mathrm{Fe/H}] = %.2f$' % self._fehs[indx] +'\n'+r'$[\alpha/\mathrm{Fe}] = %.3f$' % self._afes[indx],top_right=True)
        bovy_plot.bovy_text(r'$\mathrm{correlation} = %.2f$' % (numpy.corrcoef(numpy.array(self.__dict__['_%s_samples' % param1][indx]),numpy.array(self.__dict__['_%s_samples' % param2][indx]))[0,1]),bottom_left=True)
