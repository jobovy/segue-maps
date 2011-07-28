import os, os.path
import math
import numpy
from optparse import OptionParser
import pyfits
from galpy.util import bovy_plot, bovy_coords
from matplotlib import pyplot
import segueSelect
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
    bins, specbins= 101, 31
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
                        ec='g',ls='dashed',histtype='step')
    bovy_plot.bovy_hist(ksconst_faint,range=range,bins=bins,overplot=True,
                        ec='r',ls='solid',histtype='step')
    bovy_plot.bovy_hist(ksr_faint,range=range,bins=bins,overplot=True,
                        ec='orange',ls='solid',histtype='step')
    bovy_plot.bovy_hist(ksplatesn_r_faint,
                        range=range,bins=bins,overplot=True,
                        ec='g',ls='solid',histtype='step')
    xlegend, ylegend, dy= 0.55, 140., -10.
    bovy_plot.bovy_text(xlegend,ylegend,r'$\mathrm{constant}$',color='r')
    bovy_plot.bovy_text(xlegend,ylegend+dy,r'$r\ \mathrm{dependent}$',color='orange')
    bovy_plot.bovy_text(xlegend,ylegend+2.*dy,r'$\mathrm{plateSN\_r},r\ \mathrm{dependent}$',color='g')
    xlegend, ylegend, dy= 0.55, 105., -10.
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
    return parser

if __name__ == '__main__':
    selectFigs(get_options())
