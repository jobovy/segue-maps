import os, os.path
import math
import numpy
from optparse import OptionParser
from galpy.util import bovy_plot
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

def plot_soner(options,args):
    """Plot the r dependence of the selection function"""
    sf= segueSelect.segueSelect(sn=True,sample=options.sample,
                                plates=None,type_bright='r',
                                type_faint='r',select='all',
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
    #This is just to get rmin and rmax consistently
    allsf= segueSelect.segueSelect(sn=True,sample=options.sample,
                                plates=None,type_faint='constant')
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
    #Establish selection function for each bin, plot
    bovy_plot.bovy_print()
    if options.faint:
        if options.sample.lower() == 'k':
            yrange=[0.,2.5]
        else:
            yrange= [0.,3.5]
        bovy_plot.bovy_plot([0.,0.],[0.,0.],
                            xrange=[17.7,allsf.rmax+0.1],
                            yrange=yrange,
                            xlabel=r'$r_0\ [\mathrm{mag}]$',
                            ylabel= r'$r\ \mathrm{dependence\ of\ selection\ function}$')
    else:
        bovy_plot.bovy_plot([0.,0.],[0.,0.],
                            xrange=[allsf.rmin-0.1,17.9],
                            yrange=[0.,3.5],
                            xlabel=r'$r_0\ [\mathrm{mag}]$',
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
                                        dr_faint=0.2)
        else:
            sf= segueSelect.segueSelect(sn=True,sample=options.sample,
                                        plates=theseplates,
                                        type_bright='r',type_faint='constant',
                                        dr_bright=0.2,robust_bright=True)
        #Plot
        sf.plot_s_one_r(theseplates[0],color=bincolors[bb],
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
    sf= segueSelect.segueSelect(sn=False,sample=options.sample,
                                plates=None,select='all')
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
                            xlabel=r'$r_0\ [\mathrm{mag}]$',
                            ylabel=r'$S/N$')
    else:
        bovy_plot.bovy_plot([0.,0.],[0.,0.],
                            xrange=[sf.rmin-0.1,17.9],
                            yrange=[0.,150.],
                            xlabel=r'$r_0\ [\mathrm{mag}]$',
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
        bovy_plot.bovy_plot(sf.spec[plotindx].dered_r,
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
    return parser

if __name__ == '__main__':
    selectFigs(get_options())
