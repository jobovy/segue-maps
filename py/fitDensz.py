import sys
import os, os.path
import re
import math
import numpy
import scipy
from scipy import interpolate
import cPickle as pickle
from matplotlib import pyplot
from optparse import OptionParser
from scipy import optimize, special, integrate
import pyfits
from galpy.util import bovy_coords, bovy_plot, bovy_quadpack
import bovy_mcmc
import markovpy
from segueSelect import ivezic_dist_gr, segueSelect, _gi_gr, _mr_gi, \
    _SEGUESELECTDIR
from fitSigz import readData, _ARICHFEHRANGE, _APOORFEHRANGE
from plotData import plotDensz
#Scipy version
try:
    sversion=re.split(r'\.',scipy.__version__)
    _SCIPYVERSION=float(sversion[0])+float(sversion[1])/10.
except:
    raise ImportError( "scipy.__version__ not understood, contact developer, send scipy.__version__")
_ERASESTR= "                                                                                "
_VERBOSE=True
_DEBUG=False
_INTEGRATEPLATESEP= True #DON'T CHANGE THIS
_EPSREL= 1.45e-08
_EPSABS= 1.45e-08
_DEGTORAD=math.pi/180.
_ZSUN=0.025 #Sun's offset from the plane toward the NGP in kpc
_DZ=6.
_DR=5.
_NDS= 1001
_NGR= 11
_NFEH= 11
def fitDensz(parser):
    (options,args)= parser.parse_args()
    if len(args) == 0:
        parser.print_help()
        return
    if options.model.lower() == 'hwr':
        densfunc= _HWRDensity
    elif options.model.lower() == 'dblexp':
        densfunc= _DblExpDensity
    elif options.model.lower() == 'flare':
        densfunc= _FlareDensity
    elif options.model.lower() == 'tiedflare':
        densfunc= _TiedFlareDensity
    elif options.model.lower() == 'twovertical':
        densfunc= _TwoVerticalDensity
    elif options.model.lower() == 'twodblexp':
        densfunc= _TwoDblExpDensity
    elif options.model.lower() == 'threedblexp':
        densfunc= _ThreeDblExpDensity
    elif options.model.lower() == 'twodblexpflare':
        densfunc= _TwoDblExpFlareDensity
    if options.metal.lower() == 'rich':
        feh= -0.15
        fehrange= _APOORFEHRANGE
    elif options.metal.lower() == 'poor':
        feh= -0.65
        fehrange= _ARICHFEHRANGE
    elif options.metal.lower() == 'poorpoor':
        feh= -0.7
        fehrange= [_ARICHFEHRANGE[0],-0.7]
    elif options.metal.lower() == 'poorrich':
        feh= -0.7
        fehrange= [-0.7,_ARICHFEHRANGE[1]]
    elif options.metal.lower() == 'richpoor':
        feh= -0.3
        fehrange= [_APOORFEHRANGE[0],-0.25]
    elif options.metal.lower() == 'richrich':
        feh= -0.15
        fehrange= [-0.25,_APOORFEHRANGE[1]]
    elif options.metal.lower() == 'richpoorest':
        feh= -0.65
        fehrange= [-1.5,_APOORFEHRANGE[0]]
    elif options.metal.lower() == 'apoorpoor' or options.metal.lower() == 'apoorrich' \
                or options.metal.lower() == 'arichpoor' or options.metal.lower() == 'arichrich':
        feh= -0.65
        fehrange= [-1.5,_APOORFEHRANGE[1]]    
    else:
        feh= -0.5 
    #First read the data
    if _VERBOSE:
        print "Reading and parsing data ..."
    if options.fake:
        fakefile= open(options.fakefile,'rb')
        fakedata= pickle.load(fakefile)
        fakefile.close()
        #Calculate distance
        ds, ls, bs, rs, grs, fehs= [], [], [], [], [], []
        for ii in range(len(fakedata)):
            ds.append(_ivezic_dist(fakedata[ii][1],fakedata[ii][0],fakedata[ii][2]))
            ls.append(fakedata[ii][3])
            bs.append(fakedata[ii][4])
            rs.append(fakedata[ii][0])
            grs.append(fakedata[ii][1])
            fehs.append(fakedata[ii][2])
        ds= numpy.array(ds)
        ls= numpy.array(ls)
        bs= numpy.array(bs)
        rs= numpy.array(rs)
        grs= numpy.array(grs)
        fehs= numpy.array(fehs)
        XYZ= bovy_coords.lbd_to_XYZ(ls,bs,ds,degree=True)                      
    else:
        XYZ,vxvyvz,cov_vxvyvz,rawdata= readData(metal=options.metal,
                                                sample=options.sample,
                                                loggmin=options.loggmin,
                                                snmin=options.snmin,
                                                select=options.select)
        grs= rawdata.dered_g-rawdata.dered_r
    #Load model distributions
    if options.sample.lower() == 'g':
        colorrange=[0.48,0.55]
    elif options.sample.lower() == 'k':
        colorrange=[0.55,0.75]
    #FeH
    fehdist= DistSpline(*numpy.histogram(rawdata.feh,bins=11,range=fehrange),
                         xrange=fehrange)
    #Color
    if options.colordist.lower() == 'constant':
        colordist= _const_colordist
    elif options.colordist.lower() == 'binned':
        #Bin the colors
        nbins= 16
        hist,edges= numpy.histogram(grs,range=[grmin,grmax],bins=nbins)
        colordist= DistBinned(hist,edges)
    elif options.colordist.lower() == 'spline':
        colordist= DistSpline(*numpy.histogram(rawdata.dered_g-rawdata.dered_r,
                                           bins=9,range=colorrange),
                           xrange=colorrange)
    #Only use north or south plates?
    if options.north or options.south:
        segueplatestr= pyfits.getdata(os.path.join(_SEGUESELECTDIR,
                                                   'segueplates_ksg.fits'))
        platelb= bovy_coords.radec_to_lb(segueplatestr.ra,segueplatestr.dec,
                                         degree=True)
        indx= []
        for ii in range(len(segueplatestr.ra)):
            if options.north and platelb[ii,1] > 0.: indx.append(True)
            elif options.south and platelb[ii,1] < 0.: indx.append(True)
            else: indx.append(False)
        indx= numpy.array(indx,dtype='bool')
        plates= segueplatestr.plate[indx]
        segueplatestr= segueplatestr[indx]
        #Data
        dataindx= []
        for ii in range(len(XYZ[:,0])):
            if rawdata[ii].plate in plates: dataindx.append(True)
            else: dataindx.append(False)
        dataindx= numpy.array(dataindx,dtype='bool')
        rawdata= rawdata[dataindx]
        XYZ= XYZ[dataindx,:]
        vxvyvz= vxvyvz[dataindx,:]
        cov_vxvyvz= cov_vxvyvz[dataindx,:,:]     
    #Only consider plates with |b| > bmin?
    if not options.bmin is None or not options.bmax is None:
        segueplatestr= pyfits.getdata(os.path.join(_SEGUESELECTDIR,
                                                   'segueplates_ksg.fits'))
        platelb= bovy_coords.radec_to_lb(segueplatestr.ra,segueplatestr.dec,
                                         degree=True)
        indx= []
        for ii in range(len(segueplatestr.ra)):
            if not options.bmin is None \
                    and numpy.fabs(platelb[ii,1]) > options.bmin: 
                indx.append(True)
            elif not options.bmax is None \
                    and numpy.fabs(platelb[ii,1]) < options.bmax: 
                indx.append(True)
            else: indx.append(False)
        indx= numpy.array(indx,dtype='bool')
        plates= segueplatestr.plate[indx]
        segueplatestr= segueplatestr[indx]
        #Data
        dataindx= []
        for ii in range(len(XYZ[:,0])):
            if rawdata[ii].plate in plates: dataindx.append(True)
            else: dataindx.append(False)
        dataindx= numpy.array(dataindx,dtype='bool')
        rawdata= rawdata[dataindx]
        XYZ= XYZ[dataindx,:]
        vxvyvz= vxvyvz[dataindx,:]
        cov_vxvyvz= cov_vxvyvz[dataindx,:,:]     
    #Only consider plates around lcen bcen?
    if not options.lcen is None and not options.bcen is None \
           and not options.lbdr is None:
        from compareDataModel import similarPlatesDirection
        lbplates= similarPlatesDirection(options.lcen,options.bcen,
                                         options.lbdr,None)
        #Data
        dataindx= []
        for ii in range(len(XYZ[:,0])):
            if rawdata[ii].plate in lbplates: dataindx.append(True)
            else: dataindx.append(False)
        dataindx= numpy.array(dataindx,dtype='bool')
        rawdata= rawdata[dataindx]
        XYZ= XYZ[dataindx,:]
        vxvyvz= vxvyvz[dataindx,:]
        cov_vxvyvz= cov_vxvyvz[dataindx,:,:]     
    #Cut on platesn
    if not options.minplatesn_bright is None:
        segueplatestr= pyfits.getdata(os.path.join(_SEGUESELECTDIR,
                                                   'segueplates_ksg.fits'))
        platesn_r= (segueplatestr.sn1_1+segueplatestr.sn2_1)/2.
        indx= []
        for ii in range(len(segueplatestr.plate)):
            if not 'faint' in segueplatestr.programname[ii] \
                    and platesn_r[ii] >= options.minplatesn_bright \
                    and platesn_r[ii] <= options.maxplatesn_bright:
                indx.append(True)
            elif 'faint' in segueplatestr.programname[ii] \
                    and platesn_r[ii] >= options.minplatesn_faint \
                    and platesn_r[ii] <= options.maxplatesn_faint:
                indx.append(True)
            else:
                indx.append(False)
        indx= numpy.array(indx,dtype='bool')
        plates= segueplatestr.plate[indx]
        segueplatestr= segueplatestr[indx]
        #Data
        dataindx= []
        for ii in range(len(XYZ[:,0])):
            if rawdata[ii].plate in plates: dataindx.append(True)
            else: dataindx.append(False)
        dataindx= numpy.array(dataindx,dtype='bool')
        rawdata= rawdata[dataindx]
        XYZ= XYZ[dataindx,:]
        vxvyvz= vxvyvz[dataindx,:]
        cov_vxvyvz= cov_vxvyvz[dataindx,:,:]     
    #Cut on KS
    if not options.minks is None:
        print "WARNING: MINKS ONLY WORKS FOR G-ALL"
        segueplatestr= pyfits.getdata(os.path.join(_SEGUESELECTDIR,
                                                   'segueplates_ksg.fits'))
        if not options.minplatesn_bright is None:
            plates= segueplatestr.plate[indx]
            segueplatestr= segueplatestr[indx]
        else: plates= segueplatestr.plate
        #Cut on KS
        indx= []
        for ii in range(len(plates)):
            if options.sel_bright.lower() == 'constant' \
                    and not 'faint' in segueplatestr.programname[ii]:
                if segueplatestr.ksconst_g_all[ii] >= options.minks:
                    indx.append(True)
                else:
                    indx.append(False)
            elif options.sel_bright.lower() == 'r' \
                    and not 'faint' in segueplatestr.programname[ii]:
                if segueplatestr.ksr_g_all[ii] >= options.minks:
                    indx.append(True)
                else:
                    indx.append(False)
            elif options.sel_bright.lower() == 'platesn_r' \
                    and not 'faint' in segueplatestr.programname[ii]:
                if segueplatestr.ksplatesn_r_g_all[ii] >= options.minks:
                    indx.append(True)
                else:
                    indx.append(False)
            elif options.sel_bright.lower() == 'sharprcut' \
                    and not 'faint' in segueplatestr.programname[ii]:
                if segueplatestr.kssharp_g_all[ii] >= options.minks:
                    indx.append(True)
                else:
                    indx.append(False)
            elif options.sel_faint.lower() == 'constant' \
                    and 'faint' in segueplatestr.programname[ii]:
                if segueplatestr.ksconst_g_all[ii] >= options.minks:
                    indx.append(True)
                else:
                    indx.append(False)
            elif options.sel_faint.lower() == 'r' \
                    and 'faint' in segueplatestr.programname[ii]:
                if segueplatestr.ksr_g_all[ii] >= options.minks:
                    indx.append(True)
                else:
                    indx.append(False)
            elif options.sel_faint.lower() == 'platesn_r' \
                    and 'faint' in segueplatestr.programname[ii]:
                if segueplatestr.ksplatesn_r_g_all[ii] >= options.minks:
                    indx.append(True)
                else:
                    indx.append(False)
            elif options.sel_faint.lower() == 'sharprcut' \
                    and 'faint' in segueplatestr.programname[ii]:
                if segueplatestr.kssharp_g_all[ii] >= options.minks:
                    indx.append(True)
                else:
                    indx.append(False)
        indx= numpy.array(indx,dtype='bool')
        plates= plates[indx]
        segueplatestr= segueplatestr[indx]
        #Data
        dataindx= []
        for ii in range(len(XYZ[:,0])):
            if rawdata[ii].plate in plates: dataindx.append(True)
            else: dataindx.append(False)
        dataindx= numpy.array(dataindx,dtype='bool')
        rawdata= rawdata[dataindx]
        XYZ= XYZ[dataindx,:]
        vxvyvz= vxvyvz[dataindx,:]
        cov_vxvyvz= cov_vxvyvz[dataindx,:,:]     
    #Load selection function
    if _VERBOSE:
        print "Loading selection function ..."
    if options.fake:
        plates= None
    else:
        plates= numpy.array(list(set(list(rawdata.plate))),dtype='int') #Only load plates that we use
    if not options.bright and not options.faint and not options.cutbrightfaint:
        print "Using %i plates, %i stars ..." %(len(plates),len(XYZ[:,0]))
    sf= segueSelect(plates=plates,type_faint=options.sel_faint,
                    sample=options.sample,type_bright=options.sel_bright,
                    sn=options.snmin,select=options.select)
    if options.fake:
        plates= sf.plates
    platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                     degree=True)
    indx= [not 'faint' in name for name in sf.platestr.programname]
    platebright= numpy.array(indx,dtype='bool')
    indx= ['faint' in name for name in sf.platestr.programname]
    platefaint= numpy.array(indx,dtype='bool')
    if (options.bright or options.faint) and not options.fake:
        indx= []
        for ii in range(len(XYZ[:,0])):
            pindx= (sf.plates == rawdata[ii].plate)
            if options.bright \
                    and not 'faint' in sf.platestr[pindx].programname[0]:
                indx.append(True)
            elif options.faint \
                    and 'faint' in sf.platestr[pindx].programname[0]:
                indx.append(True)
            else:
                indx.append(False)
        indx= numpy.array(indx,dtype='bool')
        rawdata= rawdata[indx]
        XYZ= XYZ[indx,:]
        vxvyvz= vxvyvz[indx,:]
        cov_vxvyvz= cov_vxvyvz[indx,:,:]
        #Also cut the data > or < than 17.8       
        if options.bright:
            dataindx= (rawdata.dered_r < 17.8)
        elif options.faint:
            dataindx= (rawdata.dered_r >= 17.8)
        rawdata= rawdata[dataindx]
        XYZ= XYZ[dataindx,:]
        vxvyvz= vxvyvz[dataindx,:]
        cov_vxvyvz= cov_vxvyvz[dataindx,:,:]
        #Reload selection function
        plates= numpy.array(list(set(list(rawdata.plate))),dtype='int') #Only load plates that we use
        sf= segueSelect(plates=plates,type_faint=options.sel_faint,
                        type_bright=options.sel_bright,sample=options.sample,
                        sn=options.snmin,select=options.select)
        platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                         degree=True)
        indx= [not 'faint' in name for name in sf.platestr.programname]
        platebright= numpy.array(indx,dtype='bool')
        indx= ['faint' in name for name in sf.platestr.programname]
        platefaint= numpy.array(indx,dtype='bool')
    if (options.bright or options.faint) and options.fake:
        if options.bright:
            indx= (rs < 17.8)
            XYZ= XYZ[indx,:]
            plates= sf.plates[sf.brightplateindx]
        elif options.faint:
            indx= (rs >= 17.8)
            XYZ= XYZ[indx,:]
            plates= sf.plates[sf.faintplateindx]
        sf= segueSelect(plates=plates,type_faint=options.sel_faint,
                        type_bright=options.sel_bright,sample=options.sample,
                        sn=options.snmin,select=options.select)
        platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                         degree=True)
        indx= [not 'faint' in name for name in sf.platestr.programname]
        platebright= numpy.array(indx,dtype='bool')
        indx= ['faint' in name for name in sf.platestr.programname]
        platefaint= numpy.array(indx,dtype='bool')
    if options.cutbrightfaint:
        #Cut out bright stars on faint plates and vice versa
        indx= []
        for ii in range(len(data.feh)):
            if sf.platebright[str(data[ii].plate)] and data[ii].dered_r >= 17.8:
                indx.append(False)
            elif not sf.platebright[str(data[ii].plate)] and data[ii].dered_r < 17.8:
                indx.append(False)
            else:
                indx.append(True)
        indx= numpy.array(indx,dtype='bool')
        data= data[indx]
        XYZ= XYZ[indx,:]
        vxvyvz= vxvyvz[indx,:]
        cov_vxvyvz= cov_vxvyvz[indx,:]
        #Reload selection function
        plates= numpy.array(list(set(list(rawdata.plate))),dtype='int') #Only load plates that we use
        sf= segueSelect(plates=plates,type_faint=options.sel_faint,
                        type_bright=options.sel_bright,sample=options.sample,
                        sn=options.snmin,select=options.select)
        platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                         degree=True)
        indx= [not 'faint' in name for name in sf.platestr.programname]
        platebright= numpy.array(indx,dtype='bool')
        indx= ['faint' in name for name in sf.platestr.programname]
        platefaint= numpy.array(indx,dtype='bool')
    if options.bright or options.faint or options.cutbrightfaint:
        print "Using %i plates, %i stars ..." %(len(plates),len(XYZ[:,0]))
    Ap= math.pi*2.*(1.-numpy.cos(1.49*_DEGTORAD)) #SEGUE PLATE=1.49 deg radius
    if options.sample.lower() == 'g':
        grmin, grmax= 0.48, 0.55
        rmin,rmax= 14.5, 20.2
    if os.path.exists(args[0]):#Load savefile
        savefile= open(args[0],'rb')
        params= pickle.load(savefile)
        samples= pickle.load(savefile)
        savefile.close()
        print "Printing best fit parameters ..."
        print params
        print numpy.exp(params)
        print "Printing mean and std dev of samples ..."
        for ii in range(len(params)):
            xs= numpy.array([s[ii] for s in samples])
            print numpy.mean(xs), numpy.std(xs)
            print numpy.exp(numpy.mean(xs)), numpy.exp(numpy.mean(xs))-numpy.exp(numpy.mean(xs)-numpy.std(xs)), -numpy.exp(numpy.mean(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs))
        """
        print "Correlations between parameters ..."
        for ii in range(len(params)-1):
            xs= numpy.array([s[ii] for s in samples])
            for jj in range(ii+1,len(params)):
                ys= numpy.array([s[jj] for s in samples])
                print numpy.corrcoef(xs,ys,rowvar=1)
                print numpy.corrcoef(numpy.exp(xs),numpy.exp(ys),rowvar=1)
                print numpy.corrcoef(xs,numpy.exp(ys),rowvar=1)
                print numpy.corrcoef(numpy.exp(xs),ys,rowvar=1)
        """
        print "Printing 95, 99 percent lower and upper limits ..."
        for ii in range(len(params)):
            xs= sorted(numpy.array([s[ii] for s in samples]))
            indx1= int(numpy.floor(0.05*len(samples)))
            indx2= int(numpy.floor(0.01*len(samples)))
            print xs[indx1], numpy.exp(xs[indx1]), xs[indx2], numpy.exp(xs[indx2])
            indx1= int(numpy.floor(0.95*len(samples)))
            indx2= int(numpy.floor(0.99*len(samples)))
            print xs[indx1], numpy.exp(xs[indx1]), xs[indx2], numpy.exp(xs[indx2])
    else:
        #Subsample
        if not options.subsample is None:
            randindx= numpy.random.permutation(len(rawdata.ra))
            randindx= randindx[0:options.subsample]
            XYZ= XYZ[randindx,:]
            vxvyvz= vxvyvz[randindx,:]
            cov_vxvyvz= cov_vxvyvz[randindx,:,:]
            rawdata= rawdata[randindx]
        XYZ= XYZ.astype(numpy.float64)
        R= ((8.-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)
        XYZ[:,2]+= _ZSUN
        like_func= _HWRLikeMinus
        pdf_func= _HWRLike
        if options.model.lower() == 'hwr':
            if options.metal == 'rich':
                params= numpy.array([numpy.log(0.3),numpy.log(2.5),0.0])
            elif options.metal == 'poor':
                params= numpy.array([numpy.log(1.2),numpy.log(2.5),0.0])
            else:
                params= numpy.array([numpy.log(0.3),numpy.log(2.5),0.0])
            densfunc= _HWRDensity
            #Slice sampling keywords
            if options.metropolis:
                step= [0.03,0.03,0.02]
            else:
                step= [0.3,0.3,0.02]
            create_method=['step_out','step_out','step_out']
            isDomainFinite=[[False,True],[False,True],[True,True]]
            domain=[[0.,4.6051701859880918],[0.,4.6051701859880918],[0.,1.]]
        elif options.model.lower() == 'dblexp':
            if options.metal == 'rich':
                params= numpy.array([numpy.log(0.3),numpy.log(2.5)])
            elif options.metal == 'poor':
                params= numpy.array([numpy.log(1.),numpy.log(2.5)])
            else:
                params= numpy.array([numpy.log(0.3),numpy.log(2.5)])
            densfunc= _DblExpDensity
            #Slice sampling keywords
            if options.metropolis:
                step= [0.03,0.03]
            else:
                step= [0.3,0.3]
            create_method=['step_out','step_out']
            isDomainFinite=[[False,True],[False,True]]
            domain=[[0.,4.6051701859880918],[0.,4.6051701859880918]]
        elif options.model.lower() == 'flare':
            if options.metal == 'rich':
                params= numpy.array([numpy.log(0.3),numpy.log(99.5),numpy.log(5.3)])
            elif options.metal == 'poor':
                params= numpy.array([numpy.log(1.),numpy.log(2.5),numpy.log(2.5)])
            else:
                params= numpy.array([numpy.log(0.3),numpy.log(2.5),numpy.log(2.5)])
            densfunc= _FlareDensity
            #Slice sampling keywords
            if options.metropolis:
                step= [0.03,0.03,0.03]
            else:
                step= [0.3,0.3,0.3]
            create_method=['step_out','step_out','step_out']
            isDomainFinite=[[False,True],[False,True],[False,True]]
            domain=[[0.,4.6051701859880918],[0.,4.6051701859880918],
                    [0.,4.6051701859880918]]
        elif options.model.lower() == 'tiedflare':
            if options.metal == 'rich':
                params= numpy.array([numpy.log(0.3),numpy.log(2.5)])
            elif options.metal == 'poor':
                params= numpy.array([numpy.log(1.),numpy.log(2.5)])
            else:
                params= numpy.array([numpy.log(0.3),numpy.log(2.5)])
            densfunc= _TiedFlareDensity
            #Slice sampling keywords
            if options.metropolis:
                step= [0.03,0.03]
            else:
                step= [0.3,0.3]
            create_method=['step_out','step_out']
            isDomainFinite=[[False,True],[False,True]]
            domain=[[0.,4.6051701859880918],[0.,4.6051701859880918]]
        elif options.model.lower() == 'twovertical':
            if options.metal == 'rich':
                params= numpy.array([numpy.log(0.3),numpy.log(1.),numpy.log(2.5),0.025])
            elif options.metal == 'poor':
                params= numpy.array([numpy.log(1.),numpy.log(2.),numpy.log(2.5),0.025])
            else:
                params= numpy.array([numpy.log(0.3),numpy.log(1.),numpy.log(2.5),0.025])
            densfunc= _TwoVerticalDensity
            #Slice sampling keywords
            if options.metropolis:
                step= [0.03,0.03,0.03,0.025]
            else:
                step= [0.3,0.3,0.3,0.025]
            create_method=['step_out','step_out','step_out','step_out']
            isDomainFinite=[[False,True],[False,True],[False,True],[True,True]]
            domain=[[0.,4.6051701859880918],[0.,4.6051701859880918],
                    [0.,4.6051701859880918],[0.,1.]]
        elif options.model.lower() == 'twodblexp':
            if options.metal == 'rich':
                params= numpy.array([numpy.log(0.3),numpy.log(1.),numpy.log(2.5),numpy.log(2.5),0.025])
            elif options.metal == 'poor':
                params= numpy.array([numpy.log(1.),numpy.log(2.),numpy.log(2.5),numpy.log(2.5),0.025])
            else:
                params= numpy.array([numpy.log(0.3),numpy.log(1.),numpy.log(2.5),numpy.log(2.5),0.025])
            densfunc= _TwoDblExpDensity
            #Slice sampling keywords
            if options.metropolis:
                step= [0.03,0.03,0.03,0.3,0.025]
            else:
                step= [0.3,0.3,0.3,0.3,0.025]
            create_method=['step_out','step_out','step_out','step_out','step_out']
            isDomainFinite=[[False,True],[False,True],[False,True],
                            [False,True],[True,True]]
            domain=[[0.,4.6051701859880918],[0.,4.6051701859880918],
                    [0.,4.6051701859880918],
                    [0.,4.6051701859880918],[0.,1.]]
        elif options.model.lower() == 'threedblexp':
            if options.metal == 'rich':
                params= numpy.array([numpy.log(0.3),numpy.log(1.),numpy.log(0.25),numpy.log(2.5),numpy.log(2.5),numpy.log(2.5),0.025,0.025])
            elif options.metal == 'poor':
                params= numpy.array([numpy.log(.65),numpy.log(1.),numpy.log(0.3),numpy.log(2.),numpy.log(5.5),numpy.log(2.5),0.025,0.025])
            else:
                params= numpy.array([numpy.log(0.3),numpy.log(1.),numpy.log(0.8),numpy.log(2.5),numpy.log(2.5),numpy.log(2.5),0.025,0.025])
            densfunc= _ThreeDblExpDensity
            #Slice sampling keywords
            if options.metropolis:
                step= [0.03,0.03,0.03,0.03,0.03,0.3,0.025,0.025]
            else:
                step= [0.3,0.3,0.3,0.3,0.3,0.3,0.025,0.025]
            create_method=['step_out','step_out','step_out','step_out',
                           'step_out','step_out','step_out','step_out']
            isDomainFinite=[[False,True],[False,True],[False,True],
                            [False,True],[False,True],
                            [False,True],[True,True],[True,True]]
            domain=[[0.,4.6051701859880918],[0.,4.6051701859880918],
                    [0.,4.6051701859880918],[0.,4.6051701859880918],
                    [0.,4.6051701859880918],
                    [0.,4.6051701859880918],[0.,1.],[0.,1.]]
        elif options.model.lower() == 'twodblexpflare':
            if options.metal == 'rich':
                params= numpy.array([numpy.log(0.3),numpy.log(1.),numpy.log(2.5),numpy.log(2.5),numpy.log(5.),0.025])
            elif options.metal == 'poor':
                params= numpy.array([numpy.log(1.),numpy.log(2.),numpy.log(2.5),numpy.log(2.5),numpy.log(5.),0.025])
            else:
                params= numpy.array([numpy.log(0.3),numpy.log(1.),numpy.log(2.5),numpy.log(2.5),numpy.log(5.),0.025])
            densfunc= _TwoDblExpFlareDensity
            #Slice sampling keywords
            if options.metropolis:
                step= [0.03,0.03,0.03,0.3,0.3,0.025]
            else:
                step= [0.3,0.3,0.3,0.3,0.3,0.025]
            create_method=['step_out','step_out','step_out','step_out',
                           'step_out','step_out']
            isDomainFinite=[[False,True],[False,True],[False,True],
                            [False,True],[False,True],[True,True]]
            domain=[[0.,4.6051701859880918],[0.,4.6051701859880918],
                    [0.,4.6051701859880918],[0.,4.6051701859880918],
                    [0.,4.6051701859880918],[0.,1.]]
        #Integration argument based on scipy version
        usertol= (_SCIPYVERSION >= 0.9)
        #Integration grid when binning
        grs= numpy.linspace(grmin,grmax,_NGR)
        fehs= numpy.linspace(fehrange[0],fehrange[1],_NFEH)
        rhogr= numpy.array([colordist(gr) for gr in grs])
        rhofeh= numpy.array([fehdist(feh) for feh in fehs])
        mr= numpy.zeros((_NGR,_NFEH))
        for kk in range(_NGR):
            for jj in range(_NFEH):
                mr[kk,jj]= _mr_gi(_gi_gr(grs[kk]),fehs[jj])
        #if not dontbin, determine dmin and dmax
        allbright, allfaint= True, True
        if not options.dontbin:
            #dmin and dmax for this rmin, rmax
            for p in sf.plates:
                #l and b?
                pindx= (sf.plates == p)
                plateb= platelb[pindx,1][0]
                if 'faint' in sf.platestr[pindx].programname[0]:
                    allbright= False
                else:
                    allfaint= False
            if allbright:
                thisrmin, thisrmax= rmin, 17.8
            elif allfaint:
                thisrmin, thisrmax= 17.8, rmax
            else:
                thisrmin, thisrmax= rmin, rmax
            _THISNGR, _THISNFEH= 51, 51
            thisgrs= numpy.zeros((_THISNGR,_THISNFEH))
            thisfehs= numpy.zeros((_THISNGR,_THISNFEH))
            for ii in range(_THISNGR):
                if feh > -0.5: #rich, actually only starts at 0.2
                    thisfehs[ii,:]= numpy.linspace(fehrange[0],fehrange[1],
                                                   _THISNFEH)
                else:
                    thisfehs[ii,:]= numpy.linspace(fehrange[0],fehrange[1],
                                                   _THISNFEH)
            for ii in range(_THISNFEH):
                thisgrs[:,ii]= numpy.linspace(grmin,grmax,_THISNGR)
            dmin= numpy.amin(_ivezic_dist(thisgrs,thisrmin,thisfehs))
            dmax= numpy.amax(_ivezic_dist(thisgrs,thisrmax,thisfehs))
            ds= numpy.linspace(dmin,dmax,_NDS)
        else:
            dmin, dmax, ds= None, None, None
        #Optimize likelihood
        if _VERBOSE:
            print "Optimizing the likelihood ..."
        params= optimize.fmin_powell(like_func,params,
                                     args=(XYZ,R,
                                           sf,sf.plates,platelb[:,0],
                                           platelb[:,1],platebright,
                                           platefaint,Ap,
                                           grmin,grmax,rmin,rmax,
                                           fehrange[0],fehrange[1],
                                           feh,colordist,densfunc,
                                           fehdist,options.dontmargfeh,
                                           options.dontbincolorfeh,usertol,
                                           grs,fehs,rhogr,rhofeh,mr,
                                           options.dontbin,dmin,dmax,ds),
                                     callback=cb)
        #params= numpy.array([-1.34316986e+00,1.75402412e+00,5.14667706e-04])
        if _VERBOSE:
            print "Optimal likelihood:", params
        #Now sample
        if _VERBOSE:
            print "Sampling the likelihood ..."
            if options.metropolis:
                samples, faccept= bovy_mcmc.metropolis(params,
                                                       step,
                                                       pdf_func,
                                                       (XYZ,R,
                                                        sf,sf.plates,platelb[:,0],
                                                        platelb[:,1],platebright,
                                                        platefaint,Ap,
                                                        grmin,grmax,rmin,rmax,
                                                        fehrange[0],fehrange[1],
                                                        feh,colordist,densfunc,
                                                        fehdist,options.dontmargfeh,
                                                        options.dontbincolorfeh,usertol,
                                                        grs,fehs,rhogr,rhofeh,mr,
                                                        options.dontbin,dmin,dmax,ds),
                                                       symmetric=True,
                                                       nsamples=options.nsamples,
                                                       callback=cb)
                if numpy.any((faccept < 0.15)) or numpy.any((faccept > 0.6)):
                    print "WARNING: Metropolis acceptance ratio was < 0.15 or > 0.6 for a direction"
                    print "Full acceptance ratio list:"
                    print faccept                                    
            elif options.markovpy:
                #Set-up walkers
                nwalkers = 2*len(params)
                ndim     = len(params)
                nmarkovsamples= int(numpy.ceil(options.nsamples/nwalkers))
                sampler = markovpy.EnsembleSampler(nwalkers,ndim,
                                                   lambda x: pdf_func(x,
                                                                      XYZ,R,
                                                                      sf,sf.plates,platelb[:,0],
                                                                      platelb[:,1],platebright,
                                                                      platefaint,Ap,
                                                                      grmin,grmax,rmin,rmax,
                                                                      fehrange[0],fehrange[1],
                                                                      feh,colordist,densfunc,
                                                                      fehdist,options.dontmargfeh,
                                                                      options.dontbincolorfeh,usertol,
                                                                      grs,fehs,rhogr,rhofeh,mr,options.dontbin,dmin,dmax,ds),threads=1)
                #Set up initial position
                initial_position= []
                for ww in range(nwalkers):
                    thisparams= []
                    for pp in range(len(params)):
                        prop= params[pp]+numpy.random.normal()*0.01
                        if (isDomainFinite[pp][0] and prop < domain[pp][0]):
                            prop= domain[pp][0]
                        elif (isDomainFinite[pp][1] and prop > domain[pp][1]):
                            prop= domain[pp][1]
                        thisparams.append(prop)
                    initial_position.append(numpy.array(thisparams))
                #Sample
                pos, prob, state= sampler.run_mcmc(initial_position,None,nmarkovsamples)
                #Get chain
                chain= sampler.get_chain()
                samples= []
                for ww in range(nwalkers):
                    for ss in range(options.nsamples/nwalkers):
                        thisparams= []
                        for pp in range(len(params)):
                            thisparams.append(chain[ww,pp,ss])
                        samples.append(numpy.array(thisparams))
                if len(samples) > options.nsamples:
                    samples= samples[-options.nsamples:len(samples)]
            else:
                samples= bovy_mcmc.slice(params,
                                         step,
                                         pdf_func,
                                         (XYZ,R,
                                          sf,sf.plates,platelb[:,0],
                                          platelb[:,1],platebright,
                                          platefaint,Ap,
                                          grmin,grmax,rmin,rmax,
                                          fehrange[0],fehrange[1],
                                          feh,colordist,densfunc,
                                          fehdist,options.dontmargfeh,
                                          options.dontbincolorfeh,usertol,
                                          grs,fehs,rhogr,rhofeh,mr,options.dontbin,dmin,dmax,ds),
                                         create_method=create_method,
                                         isDomainFinite=isDomainFinite,
                                         domain=domain,
                                         nsamples=options.nsamples,
                                         callback=cb)
        print "Printing mean and std dev of samples ..."
        for ii in range(len(params)):
            xs= numpy.array([s[ii] for s in samples])
            print numpy.mean(xs), numpy.std(xs)
        if _VERBOSE:
            print "Saving ..."
        savefile= open(args[0],'wb')
        pickle.dump(params,savefile)
        pickle.dump(samples,savefile)
        savefile.close()
    return

    #Plot
    if options.plotzfunc:
        zs= numpy.linspace(0.3,2.,1001)
        #Plot the mean and std-dev from the posterior
        zmean= numpy.zeros(len(zs))
        nsigs= 3
        zsigs= numpy.zeros((len(zs),2*nsigs))
        fs= numpy.zeros((len(zs),len(samples)))
        for ii in range(len(samples)):
            thisparams= samples[ii]
            fs[:,ii]= numpy.log(densfunc(8.,zs,thisparams))
        #Record mean and std-devs
        zmean[:]= numpy.mean(fs,axis=1)
        norm= numpy.log(numpy.sum(numpy.exp(zmean)*(zs[1]-zs[0])))
        zmean-= norm
        if options.xmin is None or options.xmax is None:
            xrange= [numpy.amin(zs)-0.2,numpy.amax(zs)+0.1]
        else:
            xrange= [options.xmin,options.xmax]
        if options.ymin is None or options.ymax is None:
            yrange= [numpy.amin(zmean)-1.,numpy.amax(zmean)+1.]
        else:
            yrange= [options.ymin,options.ymax]
        bovy_plot.bovy_print()
        bovy_plot.bovy_plot(zs,zmean,'k-',xrange=xrange,yrange=yrange,
                            xlabel=options.xlabel,
                            ylabel=options.ylabel)
        for ii in range(nsigs):
            for jj in range(len(zs)):
                thisf= sorted(fs[jj,:])
                thiscut= 0.5*special.erfc((ii+1.)/math.sqrt(2.))
                zsigs[jj,2*ii]= thisf[int(math.ceil(thiscut*len(samples)))]
                thiscut= 1.-thiscut
                zsigs[jj,2*ii+1]= thisf[int(math.floor(thiscut*len(samples)))]
        colord, cc= (1.-0.75)/nsigs, 1
        nsigma= nsigs
        pyplot.fill_between(zs,zsigs[:,0]-norm,zsigs[:,1]-norm,color='0.75')
        while nsigma > 1:
            pyplot.fill_between(zs,zsigs[:,cc+1]-norm,zsigs[:,cc-1]-norm,
                                color='%f' % (.75+colord*cc))
            pyplot.fill_between(zs,zsigs[:,cc]-norm,zsigs[:,cc+2]-norm,
                                color='%f' % (.75+colord*cc))
            cc+= 1.
            nsigma-= 1
        bovy_plot.bovy_plot(zs,zmean,'k-',overplot=True)
        #Plot the data
        plotDensz(rawdata,sf,xrange=[zs[0],zs[-1]],normed=True,overplot=True)
        plotDensz(rawdata,sf,xrange=[zs[0],zs[-1]],normed=True,overplot=True,
                  noweights=True,color='r')
        plotDensz(rawdata,sf,xrange=[zs[0],zs[-1]],normed=True,overplot=True,
                  color='b',db=15.)
        bovy_plot.bovy_end_print(options.plotfile)
    if options.plotrfunc:
        zs= numpy.linspace(rmin,rmax,1001)
        #Plot the mean and std-dev from the posterior
        zmean= numpy.zeros(len(zs))
        thisparams= []
        for ii in range(len(params)):
            xs= numpy.array([s[ii] for s in samples])
            thisparams.append(numpy.mean(xs))
        zmean= _predict_rdist(zs,densfunc,thisparams,rmin,rmax,platelb,
                              grmin,grmax,feh,sf,colordist)
        zmean_const= _predict_rdist(zs,_ConstDensity,
                                    thisparams,rmin,rmax,platelb,
                                    grmin,grmax,feh,sf,colordist)
        if options.metal == 'rich':
            zmean_twofifty= _predict_rdist(zs,densfunc,
                                           [numpy.log(.25),
                                            thisparams[1],thisparams[2]],
                                           rmin,rmax,platelb,
                                           grmin,grmax,feh,sf,colordist)
        elif options.metal == 'poor':
            zmean_twofifty= _predict_rdist(zs,densfunc,
                                           [numpy.log(1.25),
                                            thisparams[1],thisparams[2]],
                                           rmin,rmax,platelb,
                                           grmin,grmax,feh,sf,colordist)
        """
        nsigs= 3
        zsigs= numpy.zeros((len(zs),2*nsigs))
        fs= numpy.zeros((len(zs),len(samples)))
        for ii in range(len(samples)):
            thisparams= samples[ii]
            fs[:,ii]= _predict_rdist(zs,densfunc,thisparams,rmin,rmax,platelb,
                                     grmin,grmax,feh,sf,colordist)
        #Record mean and std-devs
        zmean[:]= numpy.mean(fs,axis=1)
        """
        norm= numpy.nansum(zmean*(zs[1]-zs[0]))
        zmean/= norm
        from scipy import ndimage
        ndimage.filters.gaussian_filter1d(zmean,0.2/(zs[1]-zs[0]),output=zmean)
        norm_twofifty= numpy.nansum(zmean_twofifty*(zs[1]-zs[0]))
        zmean_twofifty/= norm_twofifty
        zmean_const/= zmean_const[0]/zmean[0]
        if options.xmin is None or options.xmax is None:
            xrange= [numpy.amin(zs)-0.2,numpy.amax(zs)+0.1]
        else:
            xrange= [options.xmin,options.xmax]
        if options.ymin is None or options.ymax is None:
            yrange= [0.,1.2*numpy.amax(zmean)]
        else:
            yrange= [options.ymin,options.ymax]
        bovy_plot.bovy_print()
        bovy_plot.bovy_plot(zs,zmean,'k-',xrange=xrange,yrange=yrange,
                            xlabel=options.xlabel,
                            ylabel=options.ylabel)
        bovy_plot.bovy_plot(zs,zmean_const,'k--',overplot=True)
        bovy_plot.bovy_plot(zs,zmean_twofifty,'r--',overplot=True)
        """
        for ii in range(nsigs):
            for jj in range(len(zs)):
                thisf= sorted(fs[jj,:])
                thiscut= 0.5*special.erfc((ii+1.)/math.sqrt(2.))
                zsigs[jj,2*ii]= thisf[int(math.ceil(thiscut*len(samples)))]
                thiscut= 1.-thiscut
                zsigs[jj,2*ii+1]= thisf[int(math.floor(thiscut*len(samples)))]
        colord, cc= (1.-0.75)/nsigs, 1
        nsigma= nsigs
        pyplot.fill_between(zs,zsigs[:,0]/norm,zsigs[:,1]/norm,color='0.75')
        while nsigma > 1:
            pyplot.fill_between(zs,zsigs[:,cc+1]/norm,zsigs[:,cc-1]/norm,
                                color='%f' % (.75+colord*cc))
            pyplot.fill_between(zs,zsigs[:,cc]/norm,zsigs[:,cc+2]/norm,
                                color='%f' % (.75+colord*cc))
            cc+= 1.
            nsigma-= 1
        bovy_plot.bovy_plot(zs,zmean,'k-',overplot=True)
        """
        #Plot the data
        hist= bovy_plot.bovy_hist(rawdata.dered_r,normed=True,bins=31,ec='k',
                                  histtype='step',
                                  overplot=True,range=xrange)
        bovy_plot.bovy_end_print(options.plotfile)



###############################################################################
#            LIKELIHOOD AND MINUS LIKELIHOOD
###############################################################################
def _HWRLike(params,XYZ,R,
             sf,plates,platel,plateb,platebright,platefaint,Ap,#selection,platelist,l,b,area of plates
             grmin,grmax,rmin,rmax,fehmin,fehmax,feh,#sample definition
             colordist,densfunc,fehdist,dontmargfeh, #function that describes the color-distribution, the density, the [Fe/H] distribution, and whether to marginalize over it
             dontbincolorfeh,usertol,
             grs,fehs,rhogr,rhofeh,mr,dontbin,dmin,dmax,ds):
    """log likelihood for the HWR model"""
    return -_HWRLikeMinus(params,XYZ,R,sf,plates,platel,plateb,platebright,
                          platefaint,Ap,
                          grmin,grmax,rmin,rmax,fehmin,fehmax,feh,
                          colordist,densfunc,fehdist,dontmargfeh,dontbincolorfeh,usertol,grs,fehs,rhogr,rhofeh,mr,dontbin,dmin,dmax,ds)

def _HWRLikeMinus(params,XYZ,R,
                  sf,plates,platel,plateb,platebright,platefaint,Ap,#selection,platelist,l,b,area of plates
                  grmin,grmax,rmin,rmax,fehmin,fehmax,feh,#sample definition
                  colordist,densfunc,fehdist,dontmargfeh, #function that describes the color-distribution and function that describes the density and that that describes the metallicity distribution, and whether to marginalize over that
                  dontbincolorfeh,usertol,
                  grs,fehs,rhogr,rhofeh,mr,dontbin,dmin,dmax,ds):
    """Minus log likelihood for all models"""
    if densfunc == _HWRDensity:
        if params[0] > 4.6051701859880918 \
                or params[1] > 4.6051701859880918 \
                or params[2] < 0. or params[2] > 1.:
            return numpy.finfo(numpy.dtype(numpy.float64)).max
    elif densfunc == _FlareDensity:
        if params[0] > 4.6051701859880918 \
                or params[1] > 4.6051701859880918 \
                or params[2] > 4.6051701859880918:
            return numpy.finfo(numpy.dtype(numpy.float64)).max       
    elif densfunc == _TiedFlareDensity:
        if params[0] > 4.6051701859880918 \
                or params[1] > 4.6051701859880918:
            return numpy.finfo(numpy.dtype(numpy.float64)).max       
    elif densfunc == _TwoVerticalDensity:
        if params[0] > 4.6051701859880918 \
                or params[1] > 4.6051701859880918 \
                or params[2] > 4.6051701859880918 \
                or params[3] < 0. or params[3] > 1.:
            return numpy.finfo(numpy.dtype(numpy.float64)).max       
    elif densfunc == _TwoDblExpDensity:
        if params[0] > 4.6051701859880918 \
                or params[1] > 4.6051701859880918 \
                or params[2] > 4.6051701859880918 \
                or params[3] > 4.6051701859880918 \
                or params[4] < 0. or params[4] > 1.:
            return numpy.finfo(numpy.dtype(numpy.float64)).max       
    elif densfunc == _ThreeDblExpDensity:
        if params[0] > 4.6051701859880918 \
                or params[1] > 4.6051701859880918 \
                or params[2] > 4.6051701859880918 \
                or params[3] > 4.6051701859880918 \
                or params[4] > 4.6051701859880918 \
                or params[5] > 4.6051701859880918 \
                or params[6] < 0. or params[6] > 1. \
                or params[7] < 0. or params[7] > 1.:
            return numpy.finfo(numpy.dtype(numpy.float64)).max       
    elif densfunc == _TwoDblExpFlareDensity:
        if params[0] > 4.6051701859880918 \
                or params[1] > 4.6051701859880918 \
                or params[2] > 4.6051701859880918 \
                or params[3] > 4.6051701859880918 \
                or params[4] > 4.6051701859880918 \
                or params[5] < 0. or params[5] > 1.:
            return numpy.finfo(numpy.dtype(numpy.float64)).max       
    #First calculate the normalizing integral
    out= _NormInt(params,XYZ,R,
                  sf,plates,platel,plateb,platebright,platefaint,Ap,
                  grmin,grmax,rmin,rmax,fehmin,fehmax,feh,
                  colordist,densfunc,fehdist,dontmargfeh,dontbincolorfeh,
                  usertol,grs,fehs,rhogr,rhofeh,mr,dontbin,dmin,dmax,ds)
    out= len(R)*numpy.log(out)
    #Then evaluate the individual densities
    out+= -numpy.sum(numpy.log(densfunc(R,XYZ[:,2],params)))
    if _DEBUG: print out, numpy.exp(params)
    return out

###############################################################################
#            NORMALIZATION INTEGRAL
###############################################################################
def _NormInt(params,XYZ,R,
             sf,plates,platel,plateb,platebright,platefaint,Ap,
             grmin,grmax,rmin,rmax,fehmin,fehmax,feh,
             colordist,densfunc,fehdist,dontmargfeh,dontbincolorfeh,usertol,
             grs,fehs,rhogr,rhofeh,mr,dontbin,dmin,dmax,ds):
    out= 0.
    if _INTEGRATEPLATESEP:
        for ii in range(len(plates)):
        #if _DEBUG: print plates[ii], sf(plates[ii])
            if sf.platebright[str(plates[ii])] and not sf.type_bright.lower() == 'sharprcut':
                thisrmin= rmin
                thisrmax= 17.8
            elif sf.platebright[str(plates[ii])] and sf.type_bright.lower() == 'sharprcut':
                thisrmin= rmin
                thisrmax= numpy.amin([sf.rcuts[str(plates[ii])],17.8])
            elif not sf.type_faint.lower() == 'sharprcut':
                thisrmin= 17.8
                thisrmax= rmax
            elif sf.type_faint.lower() == 'sharprcut':
                thisrmin= 17.8
                thisrmax= numpy.amin([sf.rcuts[str(plates[ii])],rmax])
            if dontbin and dontmargfeh and dontbincolorfeha:
                out+= bovy_quadpack.dblquad(_HWRLikeNormInt,grmin,grmax,
                                            lambda x: _ivezic_dist(x,thisrmin,feh),
                                            lambda x: _ivezic_dist(x,thisrmax,feh),
                                            args=(colordist,platel[ii],plateb[ii],
                                                  params,densfunc,sf,plates[ii],
                                                  feh),
                                            epsrel=_EPSREL,epsabs=_EPSABS)[0]
            elif dontbin and not dontmargfeh and dontbincolorfeh:
                out+= integrate.tplquad(_HWRLikeNormIntFeH,grmin,grmax,
                                        lambda x: fehmin,
                                        lambda x: fehmax,
                                        lambda x,y: _ivezic_dist(x,thisrmin,y),
                                        lambda x,y: _ivezic_dist(x,thisrmax,y),
                                        args=(colordist,platel[ii],plateb[ii],
                                              params,densfunc,sf,plates[ii],
                                              fehdist),
                                        epsrel=_EPSREL,epsabs=_EPSABS)[0]
            elif dontbin and dontmargfeh and not dontbincolorfeh:
                pass
            elif dontbin: #Marginalize over FeH by binning
                #Calculate sequence of one-d integrals
                for kk in range(_NGR):
                    for jj in range(_NFEH):
                        if usertol:
                            out+= integrate.quadrature(_HWRLikeNormIntFeH1D,
                                                       _ivezic_dist(grs[kk],
                                                                    thisrmin,
                                                                    fehs[jj]),
                                                       _ivezic_dist(grs[kk],
                                                                    thisrmax,
                                                                    fehs[jj]),
                                                       args=(mr[kk,jj],
                                                             platel[ii],
                                                             plateb[ii],params,
                                                             densfunc,sf,
                                                             plates[ii]),
                                                       tol=_EPSABS,
                                                       rtol=_EPSREL,
                                                       vec_func=True)[0]\
                                                       *rhogr[kk]*rhofeh[jj]
                        else:
                            out+= integrate.quadrature(_HWRLikeNormIntFeH1D,
                                                       _ivezic_dist(grs[kk],
                                                                    thisrmin,
                                                                    fehs[jj]),
                                                       _ivezic_dist(grs[kk],
                                                                    thisrmax,
                                                                    fehs[jj]),
                                                       args=(mr[kk,jj],
                                                             platel[ii],
                                                             plateb[ii],params,
                                                             densfunc,sf,
                                                             plates[ii]),
                                                       tol=_EPSABS,
                                                       vec_func=True)[0]\
                                                       *rhogr[kk]*rhofeh[jj]
            else: #Compute integral by binning everything
                thisout= numpy.zeros(_NDS)
                for kk in range(_NGR):
                    for jj in range(_NFEH):
                        #What rs do these ds correspond to
                        rs= 5.*numpy.log10(ds)+10.+mr[kk,jj]
                        thisout+= sf(plates[ii],r=rs)*rhogr[kk]*rhofeh[jj]
                #Calculate (R,z)s
                XYZ= bovy_coords.lbd_to_XYZ(numpy.array([platel[ii] for dd in range(len(ds))]),
                                            numpy.array([plateb[ii] for dd in range(len(ds))]),
                                            ds,degree=True)
                R= ((8.-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)
                XYZ[:,2]+= _ZSUN
                thisout*= ds**2.*densfunc(R,XYZ[:,2],params)
                out+= numpy.sum(thisout)
    else:
        print "WARNING: MARGINALIZING OVER FEH NOT IMPLEMENTED FOR GLOBAL NORMALIZATION INTEGRATION"
        #First bright plates
        brightplates= plates[platebright]
        thisrmin= rmin
        thisrmax= 17.8
        out+= bovy_quadpack.dblquad(_HWRLikeNormIntAll,grmin,grmax,
                                    lambda x: _ivezic_dist(x,thisrmin,feh),
                                    lambda x: _ivezic_dist(x,thisrmax,feh),
                                    args=(colordist,platel[platebright],
                                          plateb[platebright],
                                          params,brightplates,sf,densfunc,
                                          feh),
                                    epsrel=_EPSREL,epsabs=_EPSABS)[0]
        #then faint plates
        faintplates= plates[platefaint]
        thisrmin= 17.8
        thisrmax= rmax
        out+= bovy_quadpack.dblquad(_HWRLikeNormIntAll,grmin,grmax,
                                    lambda x: _ivezic_dist(x,thisrmin,feh),
                                    lambda x: _ivezic_dist(x,thisrmax,feh),
                                    args=(colordist,platel[platefaint],
                                          plateb[platefaint],
                                          params,faintplates,sf,densfunc,
                                          feh),
                                    epsrel=_EPSREL,epsabs=_EPSABS)[0]
    out*= Ap
    return out

def _HWRLikeNormInt(d,gr,colordist,l,b,params,densfunc,sf,plate,feh):
    #Go back to r
    mr= _mr_gi(_gi_gr(gr),feh)
    r= 5.*numpy.log10(d)+10.+mr
    select= sf(plate,r=r)
    #Color density
    rhogr= colordist(gr)
    #Spatial density
    XYZ= bovy_coords.lbd_to_XYZ(l,b,d,degree=True)
    R= ((8.-XYZ[0])**2.+XYZ[1]**2.)**(0.5)
    Z= XYZ[2]+_ZSUN
    dens= densfunc(R,Z,params)
    #Jacobian
    jac= d**2.
    return rhogr*dens*jac*select

def _HWRLikeNormIntFeH(d,feh,gr,colordist,l,b,params,densfunc,sf,plate,
                       fehdist):
    #Go back to r
    mr= _mr_gi(_gi_gr(gr),feh)
    r= 5.*numpy.log10(d)+10.+mr
    select= sf(plate,r=r)
    #Color density
    rhogr= colordist(gr)
    rhofeh= fehdist(feh)
    #Spatial density
    XYZ= bovy_coords.lbd_to_XYZ(l,b,d,degree=True)
    R= ((8.-XYZ[0])**2.+XYZ[1]**2.)**(0.5)
    Z= XYZ[2]+_ZSUN
    dens= densfunc(R,Z,params)
    #Jacobian
    jac= d**2.
    return rhogr*dens*jac*select*rhofeh

def _HWRLikeNormIntFeH1D(d,mr,l,b,params,densfunc,sf,plate):
    #Go back to r
    r= 5.*numpy.log10(d)+10.+mr
    select= sf(plate,r=r)
    #Spatial density
    XYZ= bovy_coords.lbd_to_XYZ(numpy.array([l for ii in range(len(d))]),
                                numpy.array([b for ii in range(len(d))]),
                                d,degree=True)
    #XYZ= XYZ.astype('float')
    R= ((8.-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)
    Z= XYZ[:,2]+_ZSUN
    dens= densfunc(R,Z,params)
    #Jacobian
    jac= d**2.
    #print numpy.sum(dens), numpy.sum(jac), numpy.sum(select), numpy.sum(dens*jac*select)
    return dens*jac*select

def _HWRLikeNormIntAll(d,gr,colordist,l,b,params,plates,sf,densfunc,feh):
    out= 0.
    #Go back to r
    mr= _mr_gi(_gi_gr(gr),feh)
    r= 5.*numpy.log10(d)+10.+mr
    for ii in range(len(plates)):
        #Color density
        rhogr= colordist(gr)
        #Spatial density
        XYZ= bovy_coords.lbd_to_XYZ(l[ii],b[ii],d,degree=True)
        Z= XYZ[2]+_ZSUN
        R= ((8.-XYZ[0])**2.+XYZ[1]**2.)**(0.5)
        dens= densfunc(R,Z,params)
        #Jacobian
        select= sf(plates[ii],r=r)
        jac= d**2.
        out+= rhogr*dens*jac*select
    return out

###############################################################################
#            DENSITY MODELS
###############################################################################
def _HWRDensity(R,Z,params):
    """Double exponential disk + constant,
    params= [loghz,loghR,Pbad]"""
    hR= numpy.exp(params[1])
    hz= numpy.exp(params[0])
    return ((1.-params[2])/(2.*hz*hR)\
                *numpy.exp(-(R-8.)/hR
                            -numpy.fabs(Z)/hz)\
                +params[2]/(_DZ*8.))

def _DblExpDensity(R,Z,params):
    """Double exponential disk
    params= [loghz,loghR]"""
    hR= numpy.exp(-params[1])
    return numpy.exp(-(R-8.)/hR
                      -numpy.fabs(Z)/numpy.exp(params[0]))
    
def _TwoVerticalDensity(R,Z,params):
    """Double exponential disk with two vertical scale-heights
    params= [loghz1,loghz2,loghR,Pbad]"""
    hR= numpy.exp(params[2])
    hz1= numpy.exp(params[0])
    hz2= numpy.exp(params[1])
    return numpy.exp(-(R-8.)/hR)*\
        ((1.-params[3])/hz1*numpy.exp(-numpy.fabs(Z)/hz1)
         +params[3]/hz2*numpy.exp(-numpy.fabs(Z)/hz2))

def _TwoDblExpDensity(R,Z,params):
    """Two Double exponential disks
    params= [loghz1,loghz2,loghR1,loghR2,Pbad]"""
    hR1= numpy.exp(params[2])
    hR2= numpy.exp(params[3])
    hz1= numpy.exp(params[0])
    hz2= numpy.exp(params[1])
    return ((1.-params[4])/hz1*numpy.exp(-numpy.fabs(Z)/hz1-(R-8.)/hR1)
            +params[4]/hz2*numpy.exp(-numpy.fabs(Z)/hz2-(R-8.)/hR2))

def _ThreeDblExpDensity(R,Z,params):
    """Three Double exponential disks
    params= [loghz1,loghz2,loghz3,loghR1,loghR2,loghR3,Pbad1,Pbad2]"""
    hR1= numpy.exp(params[3])
    hR2= numpy.exp(params[4])
    hR3= numpy.exp(params[5])
    hz1= numpy.exp(params[0])
    hz2= numpy.exp(params[1])
    hz3= numpy.exp(params[2])
    return ((1.-params[6]-params[7])/hz1*numpy.exp(-numpy.fabs(Z)/hz1-(R-8.)/hR1)
            +params[6]/hz2*numpy.exp(-numpy.fabs(Z)/hz2-(R-8.)/hR2)
            +params[7]/hz3*numpy.exp(-numpy.fabs(Z)/hz3-(R-8.)/hR3))

def _FlareDensity(R,Z,params):
    """Double exponential disk with flaring scale-height
    params= [loghz,loghflare,loghR]"""
    hR= numpy.exp(params[2])
    hz= numpy.exp(params[0])
    hf= hz*numpy.exp((R-8.)/numpy.exp(params[1]))
    return numpy.exp(-(R-8.)/hR)/hf*numpy.exp(-numpy.fabs(Z)/hf)

def _TwoDblExpFlareDensity(R,Z,params):
    """Two Double exponential disks, first one flares
    params= [loghz1,loghz2,loghR1,loghR2,loghf,Pbad]"""
    hR1= numpy.exp(params[2])
    hR2= numpy.exp(params[3])
    hz1= numpy.exp(params[0])
    hf= hz1*numpy.exp((R-8.)/numpy.exp(params[4]))
    hz2= numpy.exp(params[1])
    return ((1.-params[5])/hf*numpy.exp(-numpy.fabs(Z)/hf-(R-8.)/hR1)
            +params[5]/hz2*numpy.exp(-numpy.fabs(Z)/hz2-(R-8.)/hR2))

def _TiedFlareDensity(R,Z,params):
    """Double exponential disk with flaring scale-height equal to radial scale
    params= [loghz,loghR]"""
    hR= numpy.exp(params[1])
    hz= numpy.exp(params[0])
    hf= hz*numpy.exp((R-8.)/hR)
    return numpy.exp(-(R-8.)/hR)/hf*numpy.exp(-numpy.fabs(Z)/hf)
    
def _ConstDensity(R,Z,params):
    """Constant density"""
    return 1.
    
###############################################################################
#            COLOR/METALLICITY DISTRIBUTIONS
###############################################################################
def _const_colordist(gr):
    return 1./.07

class DistBinned:
    """Color distribution from a binned representation"""
    def __init__(self,hist,edges):
        self.hist= hist/(numpy.sum(hist)*(edges[1]-edges[0])) #normalized
        self.edges= edges
        return

    def __call__(self,gr):
        #Find bin that contains this gr
        bb= int(numpy.floor((gr-self.edges[0])/(self.edges[1]-self.edges[0])))
        return self.hist[bb]

class DistSpline:
    """Color distribution from a spline fit to a binned representation"""
    def __init__(self,hist,edges,xrange=None):
        self.hist= hist/(numpy.sum(hist)*(edges[1]-edges[0])) #normalized
        self.edges= edges
        xs= []
        for ii in range(len(self.hist)):
            xs.append((edges[ii]+edges[ii+1])/2.)
        self.xs= numpy.array(xs)
        self.spline= interpolate.splrep(self.xs,numpy.log(self.hist+0.00001))
        self.range= xrange
        return

    def __call__(self,gr):
        if gr > self.range[1] or gr < self.range[0]: return 0.
        return numpy.exp(interpolate.splev(gr,self.spline))

class FeHXDDist:
    """Distribution from XD for FeH"""
    def __init__(self,data,k=2,mincut=None,maxcut=None):
        from extreme_deconvolution import extreme_deconvolution
        import xdtarget
        #We assume only one of mincut,maxcut is not None
        self.mincut= mincut
        self.maxcut= maxcut
        if not mincut is None:
            thisdata= numpy.log(data-self.mincut)
        elif not maxcut is None:
            thisdata= numpy.log(self.maxcut-data)
        else:
            thisdata= data
        #Set up XD
        ydata= numpy.reshape(thisdata,(len(data),1))
        ycovar= numpy.zeros((len(thisdata),1))
        xamp= numpy.ones(k)/float(k)
        xmean= numpy.zeros((k,1))
        for kk in range(k):
            xmean[kk,:]= numpy.mean(ydata,axis=0)\
                +numpy.random.normal()*numpy.std(ydata,axis=0)
        xcovar= numpy.zeros((k,1,1))
        for kk in range(k):
            xcovar[kk,:,:]= numpy.cov(ydata.T)
        extreme_deconvolution(ydata,ycovar,xamp,xmean,xcovar)
        self.xamp= xamp
        self.xmean= xmean
        self.xcovar= xcovar
        self.xdt= xdtarget.xdtarget(amp=xamp,mean=xmean,covar=xcovar)
        return

    def __call__(self,feh):
        if not self.mincut is None:
            if feh <= self.mincut: return 0.
            thisfeh= numpy.log(feh-self.mincut)
            jac= 1./(feh-self.mincut)
        elif not self.maxcut is None:
            if feh >= self.maxcut: return 0.
            thisfeh= numpy.log(self.maxcut-feh)
            jac= 1./(self.maxcut-feh)
        else:
            thisfeh= feh
            jac= 1.
        a= numpy.reshape(numpy.array([thisfeh]),(1,1))
        acov= numpy.zeros((1,1))
        return numpy.exp(self.xdt(a,acov))[0]*jac

def cb(x): print numpy.exp(x)

def _ivezic_dist(gr,r,feh):
    d,derr= ivezic_dist_gr(gr+r,r,feh)
    return d

def get_options():
    usage = "usage: %prog [options] <savefilename>\n\nsavefilename= name of the file that the fit/samples will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("-o",dest='plotfile',
                      help="Name of file for plot")
    parser.add_option("--model",dest='model',default='HWR',
                      help="Model to fit")
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--metal",dest='metal',default='rich',
                      help="Use metal-poor or rich sample ('poor', 'rich' or 'all')")
    parser.add_option("--select",dest='select',default='all',
                      help="'all' or 'program' to select all or program stars")
    parser.add_option("--sel_bright",dest='sel_bright',default='sharprcut',
                      help="Selection function to use ('constant', 'r', 'platesn_r')")
    parser.add_option("--sel_faint",dest='sel_faint',default='sharprcut',
                      help="Selection function to use ('constant', 'r', 'platesn_r')")
    parser.add_option("--colordist",dest='colordist',default='spline',
                      help="Color distribution to use ('constant', 'binned','spline')")
    parser.add_option("--fehdist",dest='colordist',default='spline',
                      help="[Fe/H] distribution to use ('spline')")
    parser.add_option("--minplatesn_bright",dest='minplatesn_bright',type='float',
                      default=None,
                      help="If set, only consider plates with this minimal platesn_r")
    parser.add_option("--maxplatesn_bright",dest='maxplatesn_bright',type='float',
                      default=100000000000,
                      help="If set, only consider plates with this maximal platesn_r")
    parser.add_option("--minplatesn_faint",dest='minplatesn_faint',type='float',
                      default=0.,
                      help="If set, only consider plates with this minimal platesn_r")
    parser.add_option("--maxplatesn_faint",dest='maxplatesn_faint',type='float',
                      default=100000000000,
                      help="If set, only consider plates with this maximal platesn_r")
    parser.add_option("--minks",dest='minks',type='float',
                      default=None,
                      help="If set, only consider plates with this minimal spectro-photo KS value")
    parser.add_option("-n","--nsamples",dest='nsamples',type='int',
                      default=100,
                      help="Number of MCMC samples to use")
    parser.add_option("--subsample",dest='subsample',type='int',
                      default=None,
                      help="If set, use a random subset of this size instead of all of the data")
    parser.add_option("--bright",action="store_true", dest="bright",
                      default=False,
                      help="Fit just the bright plates")
    parser.add_option("--faint",action="store_true", dest="faint",
                      default=False,
                      help="Fit just the faint plates")
    parser.add_option("--fake",action="store_true", dest="fake",
                      default=False,
                      help="Data is fake")
    parser.add_option("--metropolis",action="store_true", dest="metropolis",
                      default=False,
                      help="Use metropolis sampler")
    parser.add_option("--dontmargfeh",action="store_true", dest="dontmargfeh",
                      default=False,
                      help="Don't marginalize over metallicity")
    parser.add_option("--dontbincolorfeh",action="store_true", 
                      dest="dontbincolorfeh",
                      default=False,
                      help="Don't compute the integral over color and metallicity by binning")
    parser.add_option("--dontbin",action="store_true", 
                      dest="dontbin",
                      default=False,
                      help="Don't compute the integral over color and metallicity, and r by binning")
    parser.add_option("-i",dest='fakefile',
                      help="Pickle file with the fake data")
    parser.add_option("--lcen",dest='lcen',type='float',
                      default=None,
                      help="If set, only use plates centered on this lcen (deg); use with bcen and lbdr")
    parser.add_option("--bcen",dest='bcen',type='float',
                      default=None,
                      help="If set, only use plates centered on this bcen (deg); use with lcen and lbdr")
    parser.add_option("--lbdr",dest='lbdr',type='float',
                      default=None,
                      help="If set, only use plates a distance lbdr (deg) away from lcen and bcen")
    parser.add_option("--north",action="store_true", 
                      dest="north",
                      default=False,
                      help="Only use plates with b < 0")
    parser.add_option("--south",action="store_true", 
                      dest="south",
                      default=False,
                      help="Only use plates with b > 0")
    parser.add_option("--markovpy",action="store_true", 
                      dest="markovpy",
                      default=False,
                      help="Use the MarkovPy sampler")
    parser.add_option("--bmin",dest='bmin',type='float',
                      default=None,
                      help="If set, only use plates centered on a |b| > bmin (deg0")
    parser.add_option("--bmax",dest='bmax',type='float',
                      default=None,
                      help="If set, only use plates centered on a |b| < bmin (deg0")
    parser.add_option("--loggmin",dest='loggmin',type='float',
                      default=3.75,
                      help="Minimum logg to define dwarfs")
    parser.add_option("--snmin",dest='snmin',type='float',
                      default=15.,
                      help="Minimum SN")
    parser.add_option("--cutbrightfaint",action="store_true", 
                      dest="cutbrightfaint",
                      default=False,
                      help="Cut bright stars from faint plates and vice versa")
    return parser

if __name__ == '__main__':
    fitDensz(get_options())
