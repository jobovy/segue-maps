import os, os.path
import math
import numpy
from optparse import OptionParser
import pyfits
from galpy.util import bovy_plot, bovy_coords
from matplotlib import pyplot, cm
import segueSelect
from fitDensz import _ZSUN, _DEGTORAD
from compareDataModel import _legendsize
def selectFigs(parser):
    (options,args)= parser.parse_args()
    if options.type.lower() == 'platesn':
        plot_platesn(options,args)
    elif options.type.lower() == 'snvsr':
        plot_snvsr(options,args)
    elif options.type.lower() == 'soner':
        plot_soner(options,args)
    elif options.type.lower() == 'soner_platesn':
        plot_soner_platesn(options,args)
    elif options.type.lower() == 'ks':
        plot_ks(options,args)
    elif options.type.lower() == 'colormag':
        plot_colormag(options,args)
    elif options.type.lower() == 'platesn_lb':
        plot_platesn_lb(options,args)
    elif options.type.lower() == 'ks_lb':
        plot_ks_lb(options,args)
    elif options.type.lower() == 'ks_sharpr':
        plot_ks_sharpr(options,args)
    elif options.type.lower() == 'ks_tanhr':
        plot_ks_tanhr(options,args)
    elif options.type.lower() == 'platesn_rcut':
        plot_platesn_rcut(options,args)
    elif options.type.lower() == 'sn_r_fewplates':
        plot_sn_r_fewplates(options,args)
    elif options.type.lower() == 'rcutg_rcutk':
        plot_rcutg_rcutk(options,args)
    elif options.type.lower() == 'sfrz' or options.type.lower() == 'sfxy':
        plot_sfrz(options,args)

def plot_rcutg_rcutk(options,args):
    if options.program: select= 'program'
    else: select= 'all'
    sfg= segueSelect.segueSelect(sn=True,sample='g',
                                plates=None,select=select,
                                type_bright='sharprcut',
                                type_faint='sharprcut')
    sfk= segueSelect.segueSelect(sn=True,sample='k',
                                plates=None,select=select,
                                type_bright='sharprcut',
                                type_faint='sharprcut')
    #Match sfg to sfk
    matchindxg, matchindxk= [], []
    for pp in range(len(sfg.plates)):
        matchk= (sfk.plates == sfg.plates[pp])
        if True in list(matchk): #We have a match!
            matchindxg.append(pp)
            matchindxk.append(list(matchk).index(True))
    matchindxg= numpy.array(matchindxg,dtype='int')
    matchindxk= numpy.array(matchindxk,dtype='int')
    #Now gather rcuts
    rcutsg, rcutsk, ndata, platesn= [], [], [], []
    for mm in range(len(matchindxg)):
        rcutsg.append(sfg.rcuts[str(sfg.plates[matchindxg[mm]])])
        rcutsk.append(sfk.rcuts[str(sfk.plates[matchindxk[mm]])])
        ndata.append(len(sfk.platespec[str(sfk.plates[matchindxk[mm]])]))
        platesn.append(sfk.platestr.platesn_r[matchindxk[mm]])
    rcutsg= numpy.array(rcutsg)
    rcutsk= numpy.array(rcutsk)
    ndata= numpy.array(ndata)
    platesn= numpy.array(platesn)
    platesn[(platesn > 180.)]= 180. #Saturate
    ndata= ndata/numpy.mean(ndata)*20.
    bovy_plot.bovy_print(fig_width=5./.85)  
    bovy_plot.bovy_plot(rcutsg,rcutsk,s=ndata,c=platesn,
                        cmap='jet',
                        clabel=r'$\mathrm{plateSN\_r}$',
                        ylabel=r'$\mathrm{max}\ r_K\ [\mathrm{mag}]$',
                        xlabel=r'$\mathrm{max}\ r_G\ [\mathrm{mag}]$',
                        xrange=[16.5,20.2],yrange=[16.5,20.2],
                        scatter=True,edgecolors='none',
                        colorbar=True)
    bovy_plot.bovy_plot(numpy.array([16.5,20.2]),
                        numpy.array([16.5,20.2]),ls='-',color='0.65',
                        overplot=True,zorder=-1,lw=2.)
    bovy_plot.bovy_plot(numpy.array([16.5,20.2]),
                        numpy.array([19.,19.]),ls='--',color='0.65',
                        overplot=True,zorder=-2,lw=2.)
    bovy_plot.bovy_text(16.6,19.1,r'$\mathrm{K\ star\ sample\ faint\ limit}$',
                        color='0.65')
    bovy_plot.bovy_end_print(options.plotfile)   


def plot_sn_r_fewplates(options,args):
    """Plot SN versus r_0 for a few plates"""
    numpy.random.seed(2)
    if options.program: select= 'program'
    else: select= 'all'
    sf= segueSelect.segueSelect(sn=False,sample=options.sample,
                                plates=None,select=select,
                                type_bright='sharprcut',
                                type_faint='sharprcut')
    if options.faint:
        binedges= segueSelect._BINEDGES_G_FAINT
        nbins= len(binedges)-1
        bincolors= ['%f' % (0.25 + 0.5/(nbins-1)*ii) for ii in range(nbins)]
        bincolors= ['b','g','y','r','m'] #'c' at beginning
    elif not options.faint:
        binedges= segueSelect._BINEDGES_G_BRIGHT
        nbins= len(binedges)-1
        bincolors= ['%f' % (0.25 + 0.5/(nbins-1)*ii) for ii in range(nbins)]
        bincolors= ['b','g','y','r','m'] #'c' at beginning
    _NPLATES= 2
    bovy_plot.bovy_print()
    if options.faint:
        xrange=[17.8,sf.rmax]
        yrange=[0.,50.]
        bovy_plot.bovy_plot([17.8,sf.rmax],[15.,15.],
                            'k--',
                            xrange=xrange,
                            yrange=yrange,
                            xlabel=r'$r_0\ [\mathrm{mag}]$',
                            ylabel=r'$\mathrm{SN}$')
        for bb in range(nbins):
            theseplates= []
            for ii in range(len(sf.plates)):
                plate= sf.plates[ii]
                if not options.faint and 'faint' in sf.platestr[ii].programname: continue
                elif options.faint and not 'faint' in sf.platestr[ii].programname: continue
                #What SN bin is this plate in
                kk= 0
                while kk < nbins and sf.platestr[ii].platesn_r > binedges[kk+1]:
                    kk+=1
                if kk != bb: continue
                theseplates.append(plate)
            #Now pick random plate
            p= theseplates[int(numpy.floor(numpy.random.uniform()*len(theseplates)))]
            bovy_plot.bovy_plot(sf.platespec[str(p)].dered_r,
                                sf.platespec[str(p)].sna,
                                marker='o',ls='none',mfc=bincolors[bb],ms=3,
                                mec=bincolors[bb],
                                overplot=True)
    else:
        pass
    #Legend
    if options.faint:
        xlegend, ylegend, dy= (sf.rmax-1.2/(20.3-17.8)*(sf.rmax+0.1-17.8))-0.35, yrange[1]/3.5*3.15,-(yrange[1]/3.5*.21)-1
    else:
        xlegend, ylegend, dy= 16.15, 3.15, -.21
    for ii in range(nbins-1):
        bovy_plot.bovy_text(xlegend,ylegend+dy*ii,
                            r'$%5.1f < \mathrm{plateSN\_r} \leq %5.1f$' %(binedges[ii], binedges[ii+1]),color=bincolors[ii],size=_legendsize)
    ii= nbins-1
    bovy_plot.bovy_text(xlegend,ylegend+dy*ii,
                        r'$%5.1f < \mathrm{plateSN\_r}$' %binedges[ii],
                        color=bincolors[ii],size=_legendsize)
    bovy_plot.bovy_end_print(options.plotfile)
        
def plot_platesn_rcut(options,args):
    """Plot plateSN versus rcut"""
    if options.program: select= 'program'
    else: select= 'all'
    sf= segueSelect.segueSelect(sn=True,sample=options.sample,
                                plates=None,select=select,
                                type_bright='sharprcut',
                                type_faint='sharprcut')
    if options.faint:
        xs= sf.platestr.platesn_r[sf.faintplateindx]
        plates= sf.plates[sf.faintplateindx]
        yrange=[17.8,20.2]
        xrange= [0.,160.]
    else:
        xs= sf.platestr.platesn_r[sf.brightplateindx]
        plates= sf.plates[sf.brightplateindx]
        yrange=[16.5,17.8]
        xrange= [0.,360.]
    ys= []
    for x in plates:
        ys.append(sf.rcuts[str(x)])
    ys= numpy.array(ys)
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(xs,ys,'ko',
                        xlabel=r'$\mathrm{plateSN\_r}$',
                        ylabel=r'$\mathrm{max}\ r_0\ [\mathrm{mag}]$',
                        xrange=xrange,
                        yrange=yrange,
                        ms=3)
    bovy_plot.bovy_end_print(options.plotfile)   

def plot_sfrz(options,args):
    """Plot the selection function in the R,Z plane"""
    if options.program: select= 'program'
    else: select= 'all'
    #We make this figure with the sharpr for clarity
    sf= segueSelect.segueSelect(sn=True,sample=options.sample,
                                plates=None,select=select,
                                type_bright='sharprcut',
                                type_faint='sharprcut')
    platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                     degree=True)
    feh= -0.5
    if options.sample.lower() == 'g':
        gr= (0.48+0.55)/2.
    elif options.sample.lower() == 'k':
        gr= (0.75+0.55)/2.
    if options.type.lower() == 'sfxy':
        nrs= 51
    else:
        nrs= 51
    rs= numpy.linspace(sf.rmin,sf.rmax,nrs)
    select= numpy.zeros((len(sf.plates),nrs))
    Rs= numpy.zeros((len(sf.plates),nrs))
    Zs= numpy.zeros((len(sf.plates),nrs))
    for ii in range(len(sf.plates)):
        #Calculate SF, R, and Z
        select[ii,:]= sf(sf.plates[ii],rs)
        ds, derrs= segueSelect.ivezic_dist_gr(rs+gr,rs,feh)
        XYZ= bovy_coords.lbd_to_XYZ(numpy.array([platelb[ii,0] for jj in range(len(ds))]),
                                    numpy.array([platelb[ii,1] for jj in range(len(ds))]),
                                    ds,degree=True)
        if options.type.lower() == 'sfxy':
            Rs[ii,:]= XYZ[:,0]
            Zs[ii,:]= XYZ[:,1]
        else:
            Rs[ii,:]= ((8.-XYZ[:,0])**2.+XYZ[:,1]**2.)**0.5
            Zs[ii,:]= XYZ[:,2]+_ZSUN
    #Plot all lines of sight
    select[(select == 0.)]= numpy.nan
    omin, omax= numpy.nanmin(select), numpy.nanmax(select)
    colormap = cm.jet
    plotthis= colormap(_squeeze(select,omin,omax))
    bovy_plot.bovy_print(fig_width=6.)
    if options.type.lower() == 'sfxy':
        bovy_plot.bovy_plot([100.,100.],[100.,100.],'k,',
                            xrange=[5.99,-5.99],yrange=[5.99,-5.99],
                            xlabel=r'$X\ [\mathrm{kpc}]$',
                            ylabel=r'$Y\ [\mathrm{kpc}]$')
    else:
        bovy_plot.bovy_plot([100.,100.],[100.,100.],'k,',
                            xrange=[4.,14.],yrange=[-4.,4.],
                            xlabel=r'$R\ [\mathrm{kpc}]$',
                            ylabel=r'$Z\ [\mathrm{kpc}]$')                       
    for ii in range(len(sf.plates)):
        for jj in range(nrs-1):
            if numpy.isnan(select[ii,jj]): continue
            #pyplot.plot(Rs[ii,jj],Zs[ii,jj],'o',ms=1.,
            #            mfc=plotthis[ii,jj],mec='none',mew=0.)
            pyplot.plot([Rs[ii,jj],Rs[ii,jj+1]],[Zs[ii,jj],Zs[ii,jj+1]],
                        '-',color=plotthis[ii,jj])
    #Add colorbar
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(select)
    m.set_clim(vmin=omin,vmax=omax)
    cbar= pyplot.colorbar(m,fraction=0.2)
    cbar.set_clim((omin,omax))
    cbar.set_label(r'$\mathrm{selection\ fraction}$')
    #Add arrow pointing to the Galactic Center
    from matplotlib.patches import Arrow, FancyArrowPatch
    if options.type.lower() == 'sfxy':
        xarr, dx= 4.2, 1.2
        arr= FancyArrowPatch(posA=(xarr,0.),
                             posB=(xarr+dx,0.),
                             arrowstyle='->', 
                             connectionstyle='arc3,rad=%4.2f' % (0.), 
                             shrinkA=2.0, shrinkB=2.0, mutation_scale=20.0, 
                             mutation_aspect=None,fc='k')
        #arr = Arrow(xarr,0.,dx,0., edgecolor='white',fc='k',width=0.65)
        ax = pyplot.gca()
        ax.add_patch(arr)
        bovy_plot.bovy_text(xarr+7.*dx/8.,-0.25,r'$\mathrm{GC}$',
                            size=_legendsize)
        xcen, ycen, dr, t= 8., 0., 4., 14.*_DEGTORAD
        arr= FancyArrowPatch(posA=(xcen-dr*numpy.cos(t),ycen+dr*numpy.sin(t)),
                             posB=(xcen-dr*numpy.cos(-t),ycen+dr*numpy.sin(-t)),
                             arrowstyle='<-', 
                             connectionstyle='arc3,rad=%4.2f' % (2.*t), 
                             shrinkA=2.0, shrinkB=2.0, mutation_scale=20.0, 
                             mutation_aspect=None,fc='k')
        ax.add_patch(arr)
    else:
        xarr, dx=5.5, -1.
        arr= FancyArrowPatch(posA=(xarr+0.05,0.),
                             posB=(xarr+dx*10./8.,0.),
                             arrowstyle='->', 
                             connectionstyle='arc3,rad=%4.2f' % (0.), 
                             shrinkA=2.0, shrinkB=2.0, mutation_scale=20.0, 
                             mutation_aspect=None,fc='k')
        ax = pyplot.gca()
        ax.add_patch(arr)
        bovy_plot.bovy_text(xarr+7.*dx/8.,-0.45,r'$\mathrm{GC}$',
                            size=_legendsize)
        arr= FancyArrowPatch(posA=(5.5,-0.05),
                             posB=(5.5,1.),
                             arrowstyle='->', 
                             connectionstyle='arc3,rad=%4.2f' % (0.), 
                             shrinkA=2.0, shrinkB=2.0, mutation_scale=20.0, 
                             mutation_aspect=None,fc='k')
        ax = pyplot.gca()
        ax.add_patch(arr)
        bovy_plot.bovy_text(5.55,0.2,r'$\mathrm{NGP}$',
                            size=_legendsize)
    bovy_plot.bovy_end_print(options.plotfile)

def _squeeze(o,omin,omax):
    return (o-omin)/(omax-omin)

def plot_ks_lb(options,args):
    """Plot KS value vs. lb"""
    segueplatestr= segueSelect._load_fits(os.path.join(segueSelect._SEGUESELECTDIR,
                                                       'segueplates_ksg.fits'))
    #Plot
    if options.bright:
        #select bright plates only
        brightplateindx= numpy.array([not 'faint' in segueplatestr[ii].programname \
                                          for ii in range(len(segueplatestr))],
                                     dtype='bool')
        segueplatestr= segueplatestr[brightplateindx]
        if options.sel_bright.lower() == 'constant':
            plotthis= segueplatestr.ksconst_g_all
        elif options.sel_bright.lower() == 'r':
            plotthis= segueplatestr.ksr_g_all
        if options.sel_bright.lower() == 'platesn_r':
            plotthis= segueplatestr.ksplatesn_r_g_all
    else:
        #select faint plates only
        faintplateindx= numpy.array(['faint' in segueplatestr[ii].programname \
                                         for ii in range(len(segueplatestr))],
                                    dtype='bool')
        segueplatestr= segueplatestr[faintplateindx]
        if options.sel_faint.lower() == 'constant':
            plotthis= segueplatestr.ksconst_g_all
        elif options.sel_faint.lower() == 'r':
            plotthis= segueplatestr.ksr_g_all
        if options.sel_faint.lower() == 'platesn_r':
            plotthis= segueplatestr.ksplatesn_r_g_all
    indx= (plotthis >= 0.)
    segueplatestr= segueplatestr[indx]
    plotthis= plotthis[indx]
    crange=[0.,numpy.amax(plotthis)]
    platelb= bovy_coords.radec_to_lb(segueplatestr.ra,segueplatestr.dec,
                                     degree=True)
    cmap= pyplot.cm.jet
    bovy_plot.bovy_print(fig_width=10)
    bovy_plot.bovy_plot(platelb[:,0],platelb[:,1],scatter=True,colorbar=True,
                        c=plotthis,xlabel=r'$l\ [\mathrm{deg}]$',
                        ylabel=r'$b\ [\mathrm{deg}]$',cmap=cmap,
                        clabel=r'$\mathrm{KS\ probability}$',
                        crange=crange,
                        xrange=[0.,360.],yrange=[-90.,90.])
    bovy_plot.bovy_end_print(options.plotfile)

def plot_ks_sharpr(options,args):
    """Plot KS of sharpr selection function"""
    segueplatestr= segueSelect._load_fits(os.path.join(segueSelect._SEGUESELECTDIR,
                                                       'segueplates_ksg.fits'))
    #Plot
    #select bright plates only
    brightplateindx= numpy.array([not 'faint' in segueplatestr[ii].programname \
                                      for ii in range(len(segueplatestr))],
                                 dtype='bool')
    thissegueplatestr= segueplatestr[brightplateindx]
    plotthis= thissegueplatestr.kssharp_g_all
    bins=21
    xrange= [(0.001-1./bins)/(1.+1./bins),1.]
    bovy_plot.bovy_print()
    bovy_plot.bovy_hist(plotthis,range=xrange,bins=bins,
                        xlabel=r'$\mathrm{KS\ probability\ that\ the\ spectroscopic\ sample}$'+'\n'+r'$\mathrm{was\ drawn\ from\ the\ photometric\ sample}$'+'\n'+r'$\times\ \mathrm{the\ model\ selection\ function}$',
                        ylabel=r'$\mathrm{Number\ of\ plates}$',
                        ec='k',ls='dashed',histtype='step')
    #select faint plates only
    faintplateindx= numpy.array(['faint' in segueplatestr[ii].programname \
                                     for ii in range(len(segueplatestr))],
                                dtype='bool')
    thissegueplatestr= segueplatestr[faintplateindx]
    plotthis= thissegueplatestr.kssharp_g_all
    bovy_plot.bovy_hist(plotthis,range=xrange,bins=bins,overplot=True,
                        ec='k',histtype='step')
    xlegend, ylegend, dy= 0.75, 40., -2.5
    bovy_plot.bovy_plot([xlegend-0.2,xlegend-0.1],[ylegend,ylegend],'k--',
                        overplot=True)
    bovy_plot.bovy_text(xlegend,ylegend,r'$\mathrm{bright}$')
    bovy_plot.bovy_plot([xlegend-0.2,xlegend-0.1],[ylegend+dy,ylegend+dy],'k-',
                        overplot=True)
    bovy_plot.bovy_text(xlegend,ylegend+dy,r'$\mathrm{faint}$')
    bovy_plot.bovy_end_print(options.plotfile)

def plot_ks_tanhr(options,args):
    """Plot KS of tanhr selection function"""
    segueplatestr= segueSelect._load_fits(os.path.join(segueSelect._SEGUESELECTDIR,
                                                       'segueplates_ksg.fits'))
    #Plot
    #select bright plates only
    brightplateindx= numpy.array([not 'faint' in segueplatestr[ii].programname \
                                      for ii in range(len(segueplatestr))],
                                 dtype='bool')
    thissegueplatestr= segueplatestr[brightplateindx]
    plotthis= thissegueplatestr.kstanh_g_all
    bins=16
    xrange= [-(1./bins-0.001)/(1.-1./bins),1.]
    bovy_plot.bovy_print()
    bovy_plot.bovy_hist(plotthis,range=xrange,bins=bins,
                        xlabel=r'$\mathrm{KS\ probability\ that\ the\ spectroscopic\ sample}$'+'\n'+r'$\mathrm{was\ drawn\ from\ the\ photometric\ sample}$'+'\n'+r'$\times\ \mathrm{the\ model\ selection\ function}$',
                        ylabel=r'$\mathrm{Number\ of\ plates}$',
                        ec='k',ls='dashed',histtype='step')
    #select faint plates only
    faintplateindx= numpy.array(['faint' in segueplatestr[ii].programname \
                                     for ii in range(len(segueplatestr))],
                                dtype='bool')
    thissegueplatestr= segueplatestr[faintplateindx]
    plotthis= thissegueplatestr.kstanh_g_all
    bovy_plot.bovy_hist(plotthis,range=xrange,bins=bins,overplot=True,
                        ec='k',histtype='step')
    xlegend, ylegend, dy= 0.75, 27., -1.5
    bovy_plot.bovy_plot([xlegend-0.2,xlegend-0.1],[ylegend,ylegend],'k--',
                        overplot=True)
    bovy_plot.bovy_text(xlegend,ylegend,r'$\mathrm{bright}$')
    bovy_plot.bovy_plot([xlegend-0.2,xlegend-0.1],[ylegend+dy,ylegend+dy],'k-',
                        overplot=True)
    bovy_plot.bovy_text(xlegend,ylegend+dy,r'$\mathrm{faint}$')
    bovy_plot.bovy_end_print(options.plotfile)

def plot_platesn_lb(options,args):
    """Plot platesn vs ls"""
    platestr= segueSelect._load_fits(os.path.join(segueSelect._SEGUESELECTDIR,
                                                  'segueplates_ksg.fits'))
    if options.bright:
        #select bright plates only
        brightplateindx= numpy.array([not 'faint' in platestr[ii].programname \
                                          for ii in range(len(platestr))],
                                     dtype='bool')
        platestr= platestr[brightplateindx]
        platesn_r= (platestr.sn1_1+platestr.sn2_1)/2.
        crange=[numpy.amin(platesn_r),350]
    else:
        #select faint plates only
        faintplateindx= numpy.array(['faint' in platestr[ii].programname \
                                         for ii in range(len(platestr))],
                                    dtype='bool')
        platestr= platestr[faintplateindx]
        platesn_r= (platestr.sn1_1+platestr.sn2_1)/2.
        crange=[numpy.amin(platesn_r),100]
    platelb= bovy_coords.radec_to_lb(platestr.ra,platestr.dec,
                                     degree=True)
    cmap= pyplot.cm.jet
    bovy_plot.bovy_print(fig_width=10)
    bovy_plot.bovy_plot(platelb[:,0],platelb[:,1],scatter=True,colorbar=True,
                        c=platesn_r,xlabel=r'$l\ [\mathrm{deg}]$',
                        ylabel=r'$b\ [\mathrm{deg}]$',cmap=cmap,
                        clabel=r'$\mathrm{plateSN\_r}$',
                        crange=crange,
                        xrange=[0.,360.],yrange=[-90.,90.])
    bovy_plot.bovy_end_print(options.plotfile)
   
def plot_colormag(options,args):
    """Plot the sample in color-magnitude space"""
    if options.program: select= 'program'
    else: select= 'all'
    sf= segueSelect.segueSelect(sample=options.sample,
                                type_bright='constant',
                                type_faint='constant',select=select,
                                sn=options.sn)
    bins, specbins= 51, 16
    bovy_plot.bovy_print()
    sf.plotColorMag(spec=True,bins=bins,specbins=specbins)
    bovy_plot.bovy_end_print(options.plotfile)    

def plot_ks(options,args):
    """Calculate and histogram the KS test for each plate"""
    if options.program: select= 'program'
    else: select= 'all'
    plates= None #'>2800' #For testing
    sfconst= segueSelect.segueSelect(sn=True,sample=options.sample,
                                     plates=plates,type_bright='constant',
                                     type_faint='constant',select=select)
    sfr= segueSelect.segueSelect(sn=True,sample=options.sample,
                                 plates=plates,type_bright='r',
                                 type_faint='r',select=select,
                                 dr_bright=0.05,dr_faint=0.2,
                                 robust_bright=True)
    if options.sample.lower() == 'k' and options.program:
        dr_bright= 0.4
        dr_faint= 0.5
    else:
        dr_bright= 0.2
        dr_faint= 0.2
    sfplatesn_r= segueSelect.segueSelect(sn=True,sample=options.sample,
                                         plates=plates,type_bright='platesn_r',
                                         type_faint='platesn_r',select=select,
                                         dr_bright=dr_bright,
                                         dr_faint=dr_faint,
                                         robust_bright=True)
    sfsharp= segueSelect.segueSelect(sn=True,sample=options.sample,
                                     plates=plates,type_bright='sharprcut',
                                     type_faint='sharprcut',select=select)
    sftanh= segueSelect.segueSelect(sn=True,sample=options.sample,
                                    plates=plates,type_bright='tanhrcut',
                                     type_faint='tanhrcut',select=select)
    print "constant, bright"
    ksconst_bright= sfconst.check_consistency('bright')
    print "constant, faint"
    ksconst_faint= sfconst.check_consistency('faint')
    print "r, bright"
    ksr_bright= sfr.check_consistency('bright')
    print "r, faint"
    ksr_faint= sfr.check_consistency('faint')
    print "platesnr, bright"
    ksplatesn_r_bright= sfplatesn_r.check_consistency('bright')
    print "platesnr, faint"
    ksplatesn_r_faint= sfplatesn_r.check_consistency('faint')
    print "sharprcut, bright"
    kssharp_bright= sfsharp.check_consistency('bright')
    print "sharprcut, faint"
    kssharp_faint= sfsharp.check_consistency('faint')
    print "tanhrcut, bright"
    kstanh_bright= sftanh.check_consistency('bright')
    print "tanhrcut, faint"
    kstanh_faint= sftanh.check_consistency('faint')
    #Plot
    bins=21
    range= [(0.001-1./bins)/(1.+1./bins),1.]
    bovy_plot.bovy_print()
    bovy_plot.bovy_hist(ksconst_bright,range=range,bins=bins,
                        xlabel=r'$\mathrm{KS\ probability\ that\ the\ spectroscopic\ sample}$'+'\n'+r'$\mathrm{was\ drawn\ from\ the\ photometric\ sample}$'+'\n'+r'$\times\ \mathrm{the\ model\ selection\ function}$',
                        ylabel=r'$\mathrm{Number\ of\ plates}$',
                        ec='r',ls='dashed',histtype='step')
    bovy_plot.bovy_hist(ksr_bright,range=range,bins=bins,overplot=True,
                        ec='orange',ls='dashed',histtype='step')
    bovy_plot.bovy_hist(ksplatesn_r_bright,
                        range=range,bins=bins,overplot=True,
                        ec='y',ls='dashed',histtype='step')
    bovy_plot.bovy_hist(kssharp_bright,
                        range=range,bins=bins,overplot=True,
                        ec='g',ls='dashed',histtype='step')
    bovy_plot.bovy_hist(kstanh_bright,
                        range=range,bins=bins,overplot=True,
                        ec='b',ls='dashed',histtype='step')
    bovy_plot.bovy_hist(ksconst_faint,range=range,bins=bins,overplot=True,
                        ec='r',ls='solid',histtype='step')
    bovy_plot.bovy_hist(ksr_faint,range=range,bins=bins,overplot=True,
                        ec='orange',ls='solid',histtype='step')
    bovy_plot.bovy_hist(ksplatesn_r_faint,
                        range=range,bins=bins,overplot=True,
                        ec='y',ls='solid',histtype='step')
    bovy_plot.bovy_hist(kssharp_faint,
                        range=range,bins=bins,overplot=True,
                        ec='g',ls='solid',histtype='step')
    bovy_plot.bovy_hist(kstanh_faint,
                        range=range,bins=bins,overplot=True,
                        ec='b',ls='solid',histtype='step')
    xlegend, ylegend, dy= 0.55, 140., -10.
    bovy_plot.bovy_text(xlegend,ylegend,r'$\mathrm{constant}$',color='r')
    bovy_plot.bovy_text(xlegend,ylegend+dy,r'$r\ \mathrm{dependent}$',color='orange')
    bovy_plot.bovy_text(xlegend,ylegend+2.*dy,r'$\mathrm{plateSN\_r},r\ \mathrm{dependent}$',color='y')
    bovy_plot.bovy_text(xlegend,ylegend+3.*dy,r'$\mathrm{sharp}\ r\ \mathrm{cut}$',color='g')
    bovy_plot.bovy_text(xlegend,ylegend+4.*dy,r'$\mathrm{tanh}\ r\ \mathrm{cut}$',color='b')
    xlegend, ylegend, dy= 0.55, 85., -10.
    bovy_plot.bovy_plot([xlegend-0.2,xlegend-0.1],[ylegend,ylegend],'k--',
                        overplot=True)
    bovy_plot.bovy_text(xlegend,ylegend,r'$\mathrm{bright}$')
    bovy_plot.bovy_plot([xlegend-0.2,xlegend-0.1],[ylegend+dy,ylegend+dy],'k-',
                        overplot=True)
    bovy_plot.bovy_text(xlegend,ylegend+dy,r'$\mathrm{faint}$')
    bovy_plot.bovy_end_print(options.plotfile)


def plot_soner(options,args):
    """Plot the r dependence of the selection function"""
    if options.program: select= 'program'
    else: select= 'all'
    sf= segueSelect.segueSelect(sn=True,sample=options.sample,
                                plates=None,type_bright='r',
                                type_faint='r',select=select,
                                dr_bright=0.05,dr_faint=0.2,robust_bright=True)
    if options.sample.lower() == 'k':
        yrange= [0.,2.]
    else:
        yrange= None
    bovy_plot.bovy_print()
    sf.plot_s_one_r('a faint plate',overplot=False,yrange=yrange)
    sf.plot_s_one_r('a bright plate',overplot=True)
    bovy_plot.bovy_end_print(options.plotfile)

def plot_soner_platesn(options,args):
    """Plot the r dependence of the selection function as a function of 
    plateSN"""
    if options.program: select= 'program'
    else: select= 'all'
    #This is just to get rmin and rmax consistently
    allsf= segueSelect.segueSelect(sn=True,sample=options.sample,
                                   plates=None,type_faint='constant',
                                   select=select)
    #if options.sample.lower() == 'g' and options.faint:
    if options.sample.lower() == 'k' and options.program:
        dr_bright= 0.4
        dr_faint= 0.5
    else:
        dr_bright= 0.2
        dr_faint= 0.2
    if options.faint:
        binedges= segueSelect._BINEDGES_G_FAINT
        nbins= len(binedges)-1
        bincolors= ['%f' % (0.25 + 0.5/(nbins-1)*ii) for ii in range(nbins)]
        bincolors= ['b','g','y','r','m'] #'c' at beginning
    #elif options.sample.lower() == 'g' and not options.faint:
    elif not options.faint:
        binedges= segueSelect._BINEDGES_G_BRIGHT
        nbins= len(binedges)-1
        bincolors= ['%f' % (0.25 + 0.5/(nbins-1)*ii) for ii in range(nbins)]
        bincolors= ['b','g','y','r','m'] #'c' at beginning
    #Establish selection function for each bin, plot
    bovy_plot.bovy_print()
    if options.faint:
        if options.sample.lower() == 'k':
            yrange=[0.,2.5]
        else:
            yrange= [0.,4.]
        bovy_plot.bovy_plot([0.,0.],[0.,0.],
                            xrange=[17.7,allsf.rmax+0.1],
                            yrange=yrange,
                            xlabel=r'$r\ [\mathrm{mag}]$',
                            ylabel= r'$r\ \mathrm{dependence\ of\ selection\ function}$')
    else:
        bovy_plot.bovy_plot([0.,0.],[0.,0.],
                            xrange=[allsf.rmin-0.1,17.9],
                            yrange=[0.,3.5],
                            xlabel=r'$r\ [\mathrm{mag}]$',
                            ylabel= r'$r\ \mathrm{dependence\ of\ selection\ function}$')
    for bb in range(nbins):
        theseplates= []
        for ii in range(len(allsf.plates)):
            plate= allsf.plates[ii]
            if not options.faint and 'faint' in allsf.platestr[ii].programname: continue
            elif options.faint and not 'faint' in allsf.platestr[ii].programname: continue
            #What SN bin is this plate in
            kk= 0
            while kk < nbins and allsf.platestr[ii].platesn_r > binedges[kk+1]:
                kk+=1
            if kk != bb: continue
            theseplates.append(plate)
        if options.faint:
            sf= segueSelect.segueSelect(sn=True,sample=options.sample,
                                        plates=theseplates,
                                        type_bright='constant',type_faint='r',
                                        dr_bright=dr_bright,
                                        dr_faint=dr_faint,select=select)
        else:
            sf= segueSelect.segueSelect(sn=True,sample=options.sample,
                                        plates=theseplates,
                                        type_bright='r',type_faint='constant',
                                        dr_bright=dr_bright,
                                        dr_faint=dr_faint,robust_bright=True,
                                        select=select)
        #Plot
        #Find a plate with non-zero weight to plot
        pp= 0
        while sf.weight[str(theseplates[pp])] == 0.: pp+= 1
        sf.plot_s_one_r(theseplates[pp],color=bincolors[bb],
                        overplot=True)
    #Legend
    if options.faint:
        xlegend, ylegend, dy= (allsf.rmax-1.2/(20.3-17.8)*(allsf.rmax+0.1-17.8)), yrange[1]/3.5*3.15,-(yrange[1]/3.5*.21)
    else:
        xlegend, ylegend, dy= 16.15, 3.15, -.21
    for ii in range(nbins-1):
        bovy_plot.bovy_text(xlegend,ylegend+dy*ii,
                            r'$%5.1f < \mathrm{plateSN\_r} \leq %5.1f$' %(binedges[ii], binedges[ii+1]),color=bincolors[ii])
    ii= nbins-1
    bovy_plot.bovy_text(xlegend,ylegend+dy*ii,
                        r'$%5.1f < \mathrm{plateSN\_r}$' %binedges[ii],
                        color=bincolors[ii])
    bovy_plot.bovy_end_print(options.plotfile)


def plot_snvsr(options,args):
    """Plot the SN versus r for faint/bright plates and different samples"""
    if options.program: select= 'program'
    else: select= 'all'
    sf= segueSelect.segueSelect(sn=False,sample=options.sample,
                                plates=None,select=select)
    #if options.sample.lower() == 'g' and options.faint:
    if options.faint:
        binedges= segueSelect._BINEDGES_G_FAINT
        nbins= len(binedges)-1
        bincolors= ['%f' % (0.25 + 0.5/(nbins-1)*ii) for ii in range(nbins)]
        bincolors= ['b','g','y','r','m'] #'c' at beginning
    #elif options.sample.lower() == 'g' and not options.faint:
    elif not options.faint:
        binedges= segueSelect._BINEDGES_G_BRIGHT
        nbins= len(binedges)-1
        bincolors= ['%f' % (0.25 + 0.5/(nbins-1)*ii) for ii in range(nbins)]
        bincolors= ['b','g','y','r','m'] #'c' at beginning
    #Plot all plates
    bovy_plot.bovy_print()
    if options.faint:
        bovy_plot.bovy_plot([0.,0.],[0.,0.],
                            xrange=[17.7,sf.rmax+0.1],
                            yrange=[0.,50.],
                            xlabel=r'$r\ [\mathrm{mag}]$',
                            ylabel=r'$S/N$')
    else:
        bovy_plot.bovy_plot([0.,0.],[0.,0.],
                            xrange=[sf.rmin-0.1,17.9],
                            yrange=[0.,150.],
                            xlabel=r'$r\ [\mathrm{mag}]$',
                            ylabel=r'$S/N$')
    for ii in range(len(sf.plates)):
        plate= sf.plates[ii]
        if not options.faint and 'faint' in sf.platestr[ii].programname: continue
        elif options.faint and not 'faint' in sf.platestr[ii].programname: continue
        #What SN bin is this plate in
        kk= 0
        while kk < nbins and sf.platestr[ii].platesn_r > binedges[kk+1]:
            kk+=1 
        #Plot these data
        plotindx= (sf.spec.plate == plate)
        if numpy.sum(plotindx) == 0: continue
        bovy_plot.bovy_plot(sf.spec[plotindx].r,
                            sf.spec[plotindx].sna,
                            color=bincolors[kk],marker=',',ls='none',
                            overplot=True)
    bovy_plot.bovy_plot([sf.rmin-0.1,sf.rmax+0.1],[15.,15.],'k--',overplot=True)
    #Legend
    if options.faint:
        xlegend, ylegend, dy= (sf.rmax-1.2/(20.3-17.8)*(sf.rmax+0.1-17.8)), 45., -3.
    else:
        xlegend, ylegend, dy= 16.15, 135., -9.
    for ii in range(nbins-1):
        bovy_plot.bovy_text(xlegend,ylegend+dy*ii,
                            r'$%5.1f < \mathrm{plateSN\_r} \leq %5.1f$' %(binedges[ii], binedges[ii+1]),color=bincolors[ii])
    ii= nbins-1
    bovy_plot.bovy_text(xlegend,ylegend+dy*ii,
                        r'$%5.1f < \mathrm{plateSN\_r}$' %binedges[ii],
                        color=bincolors[ii])
    bovy_plot.bovy_end_print(options.plotfile)

def plot_platesn(options,args):
    """Plot the platesn vs other sns"""
    sf= segueSelect.segueSelect(sn=True,sample='G') #Unimportant
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(sf.platestr.platesn_r,sf.platestr.sn1_1,'gv',
                        xrange=[0.,300.],yrange=[0.,300],
                        xlabel=r'$\mathrm{plateSN\_r} \equiv (\mathrm{sn1\_1} + \mathrm{sn2\_1})/2$',
                        ylabel=r'$\mathrm{snX\_Y}$')
    bovy_plot.bovy_plot(sf.platestr.platesn_r,sf.platestr.sn2_1,'y^',
                        overplot=True)
    bovy_plot.bovy_plot(sf.platestr.platesn_r,sf.platestr.sn1_0,'bs',
                        overplot=True)
    bovy_plot.bovy_plot(sf.platestr.platesn_r,sf.platestr.sn2_0,'cp',
                        overplot=True)
    bovy_plot.bovy_plot(sf.platestr.platesn_r,sf.platestr.sn1_2,'rh',
                        overplot=True)
    bovy_plot.bovy_plot(sf.platestr.platesn_r,sf.platestr.sn2_2,'mH',
                        overplot=True)
    bovy_plot.bovy_text(25.,280,r'$\mathrm{sn1\_1}:\ r\ \mathrm{band}$',color='g',size=14.)
    bovy_plot.bovy_plot([15.],[285.],'gv',overplot=True)
    bovy_plot.bovy_text(25.,265,r'$\mathrm{sn2\_1}:\ r\ \mathrm{band}$',color='y',size=14.)
    bovy_plot.bovy_plot([15.],[270.],'y^',overplot=True)
    bovy_plot.bovy_text(25.,250,r'$\mathrm{sn1\_0}:\ g\ \mathrm{band}$',color='b',size=14.)
    bovy_plot.bovy_plot([15.],[255.],'bs',overplot=True)
    bovy_plot.bovy_text(25.,235,r'$\mathrm{sn2\_0}:\ g\ \mathrm{band}$',color='c',size=14.)
    bovy_plot.bovy_plot([15.],[240.],'cp',overplot=True)
    bovy_plot.bovy_text(25.,220,r'$\mathrm{sn1\_2}:\ i\ \mathrm{band}$',color='r',size=14.)
    bovy_plot.bovy_plot([15.],[225.],'rh',overplot=True)
    bovy_plot.bovy_text(25.,205,r'$\mathrm{sn2\_2}:\ i\ \mathrm{band}$',color='m',size=14.)
    bovy_plot.bovy_plot([15.],[210.],'mH',overplot=True)
    bovy_plot.bovy_end_print(options.plotfile)
    
def get_options():
    usage = "usage: %prog [options] <savefilename>\n\nsavefilename= name of the file that the fit/samples will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("-o",dest='plotfile',
                      help="Name of file for plot")
    parser.add_option("-t",dest='type',
                      help="Type of plot to make")
    parser.add_option("--sample",dest='sample',default='g',
                      help="Sample to use")
    parser.add_option("--faint",action="store_true", dest="faint",
                      default=False,
                      help="Use faint plates when a distinction between bright and faint needs to made")
    parser.add_option("--program",action="store_true", dest="program",
                      default=False,
                      help="Just use program stars")
    parser.add_option("--sn",action="store_true", dest="sn",
                      default=False,
                      help="Cut on S/N")
    parser.add_option("--bright",action="store_true", dest="bright",
                      default=False,
                      help="Use bright plates")
    parser.add_option("--sel_bright",dest='sel_bright',default='constant',
                      help="Selection function to use ('constant', 'r', 'platesn_r')")
    parser.add_option("--sel_faint",dest='sel_faint',default='platesn_r',
                      help="Selection function to use ('constant', 'r', 'platesn_r')")
    return parser

if __name__ == '__main__':
    selectFigs(get_options())
