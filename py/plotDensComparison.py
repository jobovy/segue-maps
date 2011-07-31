import os, os.path
import numpy
from optparse import OptionParser
from galpy.util import bovy_plot
import segueSelect
from fitSigz import readData
from fitDensz import _HWRDensity, _FlareDensity, _const_colordist, \
    DistSpline
import compareDataModel
def compareGRichRdist(options,args):
    if options.png: ext= 'png'
    else: ext= 'ps'
    #Set up density models and their parameters
    model1= _FlareDensity
    model2= _HWRDensity
    if options.metal.lower() == 'rich':
        params1=  numpy.array([-1.20172829533,1.01068814092,-0.0464210825653])
        params2= numpy.array([-1.34316986e+00,1.75402412e+00,5.14667706e-04])
    else:
        params1=  numpy.array([-0.187391923558,0.71285154528,1.30084421599])
        params2= numpy.array([-0.3508148171668,0.65752,0.00206572947631])
    #Load sf
    sf= segueSelect.segueSelect(sample=options.sample,sn=True,
                                type_bright='sharprcut',
                                type_faint='sharprcut')
    if options.metal.lower() == 'rich':
        feh= -0.15
        fehrange= [-0.4,0.5]
    elif options.metal.lower() == 'poor':
        feh= -0.65
        fehrange= [-1.5,-0.5]
    #Load data
    XYZ,vxvyvz,cov_vxvyvz,data= readData(metal=options.metal,
                                         sample=options.sample)
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
    if options.sample.lower() == 'g':
        colorrange=[0.48,0.55]
        rmax= 19.5 #SF cuts off here
    elif options.sample.lower() == 'k':
        colorrange=[0.55,0.75]
        rmax= 19.
    #Load model distributions
    #FeH
    fehdist= DistSpline(*numpy.histogram(data.feh,bins=11,range=fehrange),
                         xrange=fehrange)
    #Color
    cdist= DistSpline(*numpy.histogram(data.dered_g-data.dered_r,
                                       bins=9,range=colorrange),
                       xrange=colorrange)
    #We do bright/faint for 4 directions and all, all bright, all faint
    ls= [180,180,45,45]
    bs= [0,90,-23,23]
    bins= 21
    #Set up comparison
    if options.type == 'r':
        compare_func= compareDataModel.comparerdistPlate
    elif options.type == 'z':
        compare_func= compareDataModel.comparezdistPlate
    elif options.type == 'R':
        compare_func= compareDataModel.compareRdistPlate
    #all, faint, bright
    plates= ['all','bright','faint']
    for plate in plates:
        bovy_plot.bovy_print()
        compare_func(model1,params1,sf,cdist,fehdist,
                     data,plate,color='k',
                     rmin=14.5,rmax=rmax,
                     grmin=colorrange[0],
                     grmax=colorrange[1],
                     fehmin=fehrange[0],fehmax=fehrange[1],feh=feh,
                     bins=bins,ls='-')
        compare_func(model2,params2,sf,cdist,fehdist,
                     data,plate,color='k',bins=bins,
                     rmin=14.5,rmax=rmax,
                     grmin=colorrange[0],
                     grmax=colorrange[1],
                     fehmin=fehrange[0],fehmax=fehrange[1],feh=feh,
                     overplot=True,ls='--')
        if options.type == 'r':
            bovy_plot.bovy_end_print(os.path.join(args[0],'Flare_Dblexp_g_'+options.metal+'_'+plate+'.'+ext))
        else:
            bovy_plot.bovy_end_print(os.path.join(args[0],'Flare_Dblexp_g_'+options.metal+'_'+options.type+'dist_'+plate+'.'+ext))
    bins= 16
    for ii in range(len(ls)):
        #Bright
        plate= compareDataModel.similarPlatesDirection(ls[ii],bs[ii],20.,
                                                       sf,data,
                                                       faint=False)
        bovy_plot.bovy_print()
        compare_func(model1,params1,sf,cdist,fehdist,
                     data,plate,color='k',
                     rmin=14.5,rmax=rmax,
                     grmin=colorrange[0],
                     grmax=colorrange[1],
                     fehmin=fehrange[0],fehmax=fehrange[1],feh=feh,
                     bins=bins,ls='-')
        compare_func(model2,params2,sf,cdist,fehdist,
                     data,plate,color='k',bins=bins,
                     rmin=14.5,rmax=rmax,
                     grmin=colorrange[0],
                     grmax=colorrange[1],
                     fehmin=fehrange[0],fehmax=fehrange[1],feh=feh,
                     overplot=True,ls='--')
        if options.type == 'r':
            bovy_plot.bovy_end_print(os.path.join(args[0],'Flare_Dblexp_g_'+options.metal+'_l%i_b%i_bright.' % (ls[ii],bs[ii]))+ext)
        else:
            bovy_plot.bovy_end_print(os.path.join(args[0],'Flare_Dblexp_g_'+options.metal+'_'+options.type+'dist_l%i_b%i_bright.' % (ls[ii],bs[ii]))+ext)
        #Faint
        plate= compareDataModel.similarPlatesDirection(ls[ii],bs[ii],20.,
                                                       sf,data,
                                                       bright=False)
        bovy_plot.bovy_print()
        compare_func(model1,params1,sf,cdist,fehdist,
                     data,plate,color='k',
                     rmin=14.5,rmax=rmax,
                     grmin=colorrange[0],
                     grmax=colorrange[1],
                     fehmin=fehrange[0],fehmax=fehrange[1],feh=feh,
                     bins=bins,ls='-')
        compare_func(model2,params2,sf,cdist,fehdist,
                     data,plate,color='k',bins=bins,
                     rmin=14.5,rmax=rmax,grmin=colorrange[0],
                     grmax=colorrange[1],
                     fehmin=fehrange[0],fehmax=fehrange[1],feh=feh,
                     overplot=True,ls='--')
        if options.type == 'r':
            bovy_plot.bovy_end_print(os.path.join(args[0],'Flare_Dblexp_g_'+options.metal+'_l%i_b%i_faint.' % (ls[ii],bs[ii]))+ext)
        elif options.type == 'z':
            bovy_plot.bovy_end_print(os.path.join(args[0],'Flare_Dblexp_g_'+options.metal+'_'+options.type+'dist_l%i_b%i_faint.' % (ls[ii],bs[ii]))+ext)
    return None

def scatterData(options,args):
    if options.png: ext= 'png'
    else: ext= 'ps'
    #Load sf
    sf= segueSelect.segueSelect(sample=options.sample,sn=True,
                                type_bright='sharprcut',
                                type_faint='sharprcut')
    #Load data
    XYZ,vxvyvz,cov_vxvyvz,data= readData(metal=options.metal,
                                         sample=options.sample)
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
    R= ((8.-XYZ[:,0])**2.+XYZ[:,1]**2.)**0.5
    bovy_plot.bovy_print()
    if options.type.lower() == 'dataxy':
        bovy_plot.bovy_plot(XYZ[:,0],XYZ[:,1],'k,',
                            xlabel=r'$X\ [\mathrm{kpc}]$',
                            ylabel=r'$Y\ [\mathrm{kpc}]$',
                            xrange=[5,-5],yrange=[5,-5],
                            onedhists=True)
    elif options.type.lower() == 'datarz':
        bovy_plot.bovy_plot(R,XYZ[:,2],'k,',
                            xlabel=r'$R\ [\mathrm{kpc}]$',
                            ylabel=r'$Z\ [\mathrm{kpc}]$',
                            xrange=[5,14],
                            yrange=[-4,4],
                            onedhists=True)
    bovy_plot.bovy_end_print(os.path.join(args[0],options.type+'_'
                                          +options.sample+'_'+
                                          options.metal+'.'+ext))
    
def get_options():
    usage = "usage: %prog [options] <savedir>\n\nsavedir= name of the directory that the comparisons will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--metal",dest='metal',default='rich',
                      help="Use metal-poor or rich sample ('poor', 'rich' or 'all')")
    parser.add_option("-t","--type",dest='type',default='r',
                      help="Type of comparison to make ('r', 'z', 'R', 'dataxy' or 'datarz')")
    parser.add_option("--png",action="store_true", dest="png",
                      default=False,
                      help="Save as png, otherwise ps")
    return parser


if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    if options.type.lower() == 'datarz' or options.type.lower() == 'dataxy':
        scatterData(options,args)
    else:
        compareGRichRdist(options,args)
