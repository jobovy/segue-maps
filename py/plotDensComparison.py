import os, os.path
import numpy
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_plot, bovy_coords
import segueSelect
from fitSigz import readData, _ARICHFEHRANGE, _APOORFEHRANGE, \
    _ARICHAFERANGE, _APOORAFERANGE
from fitDensz import _HWRDensity, _FlareDensity, _const_colordist, \
    DistSpline, _ivezic_dist, _TwoVerticalDensity, _TwoDblExpDensity, \
    _ThreeDblExpDensity
import compareDataModel
def compareGRichRdist(options,args):
    if options.png: ext= 'png'
    else: ext= 'ps'
    #Set up density models and their parameters
    model1= _TwoDblExpDensity
    model2= _TwoDblExpDensity
    model3= _TwoDblExpDensity
    params2= None
    params3= None
    #model2= _HWRDensity
    left_legend= None
    if options.metal.lower() == 'rich':
        params1= numpy.array([-1.2773895676,-0.342622618893,
                               1.54219678317,3.26125640181,
                               0.0213463610149])
        params2= numpy.array([-1.2773895676,-0.342622618893,
                               numpy.log(2.),3.26125640181,
                               0.0213463610149])
        params3= numpy.array([-1.2773895676,-0.342622618893,
                               numpy.log(3.),3.26125640181,
                               0.0213463610149])
    elif options.metal.lower() == 'poor':
        params1= numpy.array([-0.42318463388,0.120618411932,
                               0.646532463713,1.18310796794,
                               0.0388013358113])
        params2= numpy.array([-0.42318463388,0.120618411932,
                               numpy.log(3.),1.18310796794,
                               0.0388013358113])
        params3= numpy.array([-0.42318463388,0.120618411932,
                               numpy.log(4.),1.18310796794,
                               0.0388013358113])
    elif options.metal.lower() == 'poorpoor':
        model1= _TwoDblExpDensity
        model2= _TwoDblExpDensity
        model3= _TwoDblExpDensity
        params1= numpy.array([-0.219870550136,-0.238834348878,0.669726269879,
                               0.673337061069,0.0742667712409])
        params2= numpy.array([-0.219870550136,-0.238834348878,
                               numpy.log(3.),
                               0.673337061069,0.0742667712409])
        params3= numpy.array([-0.219870550136,-0.238834348878,
                               numpy.log(4.),
                               0.673337061069,0.0742667712409])
        left_legend= r'$[\mathrm{Fe/H}] < -0.70$'
    elif options.metal.lower() == 'poorrich':
        model1= _TwoDblExpDensity
        model2= _TwoDblExpDensity
        model3= _TwoDblExpDensity
        params1= numpy.array([-0.558321132216,0.081671892084,
                               0.625197219264,1.52024378067,
                               0.0324806215411])
        params2= numpy.array([-0.558321132216,0.081671892084,
                               numpy.log(3.),1.52024378067,
                               0.0324806215411])
        params3= numpy.array([-0.558321132216,0.081671892084,
                               numpy.log(4.),1.52024378067,
                               0.0324806215411])
        left_legend= r'$[\mathrm{Fe/H}] \geq -0.70$'
    elif options.metal.lower() == 'richpoor':
        model1= _TwoDblExpDensity
        model2= _TwoDblExpDensity
        model3= _TwoDblExpDensity
        params1= numpy.array([-1.13590587909,-0.229095217496,
                               2.31957547293,3.08130976581,0.0306050071738])
        params2= numpy.array([-1.13590587909,-0.229095217496,
                               numpy.log(2.),3.08130976581,0.0306050071738])
        params3= numpy.array([-1.13590587909,-0.229095217496,
                               numpy.log(3.),3.08130976581,0.0306050071738])
        left_legend= r'$[\mathrm{Fe/H}] < -0.25$'
    elif options.metal.lower() == 'richrich':
        model1= _TwoDblExpDensity
        params1= numpy.array([-1.4054707897,-0.316827211246,
                               1.13722477602,
                               1.73361114293,0.0105419023466])
        params2= numpy.array([-1.4054707897,-0.316827211246,
                               numpy.log(2.),
                               1.73361114293,0.0105419023466])
        left_legend= r'$[\mathrm{Fe/H}] \geq -0.25$'
        params3= None
    elif options.metal.lower() == 'richpoorest':
        model1= _TwoDblExpDensity
        params1= numpy.array([-0.776456513258,0.446361154618,4.47547740363,
                               2.83176699906,0.0792706192638])
        params2= None
    elif options.metal.lower() == 'apoorpoor':
        model1= _TwoDblExpDensity
        params1= numpy.array([-1.4636093883,-0.253335318811,2.31749026865,
                               4.56846956075,0.0111069900004])
        params2= None
        left_legend= r'$[\mathrm{Fe/H}] < -0.7$'
    elif options.metal.lower() == 'apoorrich':
        model1= _TwoDblExpDensity
        params1= numpy.array([-0.950520030144,0.03112785181,1.71614632276,
                               2.58730852881,0.0316080213337])
        params2= None
        left_legend= r'$[\mathrm{Fe/H}] \geq -0.7$'
    elif options.metal.lower() == 'arichpoor':
        model1= _TwoDblExpDensity
        params1= numpy.array([-0.46348820713,-0.0330860858936,0.650016021926,
                               1.13095321369,0.060682753378])
        params2= None
    elif options.metal.lower() == 'arichrich':
        model1= _TwoDblExpDensity
        params1= numpy.array([])
        params2= None
    #Load sf
    sf= segueSelect.segueSelect(sample=options.sample,sn=True,
                                type_bright='tanhrcut',
                                type_faint='tanhrcut')
    if options.metal.lower() == 'rich':
        feh= -0.15
        fehrange= _APOORFEHRANGE
    elif options.metal.lower() == 'poor':
        feh= -0.65
        fehrange= _ARICHFEHRANGE
    elif options.metal == 'poorpoor':
        feh= -0.65
        fehrange= [_ARICHFEHRANGE[0],-0.7]
    elif options.metal == 'poorrich':
        feh= -0.65
        fehrange= [-0.7,_ARICHFEHRANGE[1]]
    elif options.metal == 'richpoor':
        feh= -0.15
        fehrange= [_APOORFEHRANGE[0],-0.25]
    elif options.metal == 'richrich':
        feh= -0.15
        fehrange= [-0.25,_APOORFEHRANGE[1]]
    elif options.metal == 'richpoorest':
        feh= -0.15
        fehrange= [-1.5,_APOORFEHRANGE[0]]
    elif options.metal == 'apoorpoor' or options.metal == 'apoorrich' \
            or options.metal == 'arichpoor' or options.metal == 'arichrich':
        feh= -0.2
        fehrange= [-1.5,_APOORFEHRANGE[1]]
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
    #Ranges
    if options.type == 'z':
        xrange= [-0.1,5.]
    elif options.type == 'R':
        xrange= [4.8,14.2]
    elif options.type == 'r':
        xrange= [14.2,20.1]
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
    if options.metal.lower() == 'poor':
        bins= [51,51,26]
    elif options.metal.lower() == 'rich':
        bins= [51,31,31]
    else:
        bins= [31,31,31]
    plates= ['all','bright','faint']
    for ii in range(len(plates)):
        plate= plates[ii]
        if plate == 'all':
            thisleft_legend= left_legend
        else:
            thisleft_legend= None
        bovy_plot.bovy_print()
        compare_func(model1,params1,sf,cdist,fehdist,
                     data,plate,color='k',
                     rmin=14.5,rmax=rmax,
                     grmin=colorrange[0],
                     grmax=colorrange[1],
                     fehmin=fehrange[0],fehmax=fehrange[1],feh=feh,
                     xrange=xrange,
                     bins=bins[ii],ls='-',left_legend=thisleft_legend)
        if not params2 is None:
            compare_func(model2,params2,sf,cdist,fehdist,
                         data,plate,color='k',bins=bins[ii],
                         rmin=14.5,rmax=rmax,
                         grmin=colorrange[0],
                         grmax=colorrange[1],
                         fehmin=fehrange[0],fehmax=fehrange[1],feh=feh,
                         xrange=xrange,
                         overplot=True,ls='--')
        if not params3 is None:
            compare_func(model3,params3,sf,cdist,fehdist,
                         data,plate,color='k',bins=bins[ii],
                         rmin=14.5,rmax=rmax,
                         grmin=colorrange[0],
                         grmax=colorrange[1],
                         fehmin=fehrange[0],fehmax=fehrange[1],feh=feh,
                         xrange=xrange,
                         overplot=True,ls=':')
        if options.type == 'r':
            bovy_plot.bovy_end_print(os.path.join(args[0],'model_data_g_'+options.metal+'_'+plate+'.'+ext))
        else:
            bovy_plot.bovy_end_print(os.path.join(args[0],'model_data_g_'+options.metal+'_'+options.type+'dist_'+plate+'.'+ext))
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
                     xrange=xrange,
                     bins=bins,ls='-')
        if not params2 is None:
            compare_func(model2,params2,sf,cdist,fehdist,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,
                         grmin=colorrange[0],
                         grmax=colorrange[1],
                         fehmin=fehrange[0],fehmax=fehrange[1],feh=feh,
                         xrange=xrange,
                         overplot=True,ls='--')
        if not params3 is None:
            compare_func(model3,params3,sf,cdist,fehdist,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,
                         grmin=colorrange[0],
                         grmax=colorrange[1],
                         fehmin=fehrange[0],fehmax=fehrange[1],feh=feh,
                         xrange=xrange,
                         overplot=True,ls=':')
        if options.type == 'r':
            bovy_plot.bovy_end_print(os.path.join(args[0],'model_data_g_'+options.metal+'_l%i_b%i_bright.' % (ls[ii],bs[ii]))+ext)
        else:
            bovy_plot.bovy_end_print(os.path.join(args[0],'model_data_g_'+options.metal+'_'+options.type+'dist_l%i_b%i_bright.' % (ls[ii],bs[ii]))+ext)
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
                     xrange=xrange,
                     bins=bins,ls='-')
        if not params2 is None:
            compare_func(model2,params2,sf,cdist,fehdist,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,grmin=colorrange[0],
                         grmax=colorrange[1],
                         fehmin=fehrange[0],fehmax=fehrange[1],feh=feh,
                         xrange=xrange,
                         overplot=True,ls='--')
        if not params3 is None:
            compare_func(model3,params3,sf,cdist,fehdist,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,grmin=colorrange[0],
                         grmax=colorrange[1],
                         fehmin=fehrange[0],fehmax=fehrange[1],feh=feh,
                         xrange=xrange,
                         overplot=True,ls=':')
        if options.type == 'r':
            bovy_plot.bovy_end_print(os.path.join(args[0],'model_data_g_'+options.metal+'_l%i_b%i_faint.' % (ls[ii],bs[ii]))+ext)
        else:
            bovy_plot.bovy_end_print(os.path.join(args[0],'model_data_g_'+options.metal+'_'+options.type+'dist_l%i_b%i_faint.' % (ls[ii],bs[ii]))+ext)
    return None

def scatterData(options,args):
    if options.png: ext= 'png'
    else: ext= 'ps'
    #Load sf
    sf= segueSelect.segueSelect(sample=options.sample,sn=True,
                                type_bright='sharprcut',
                                type_faint='sharprcut')
    if options.fake:
        fakefile= open(options.fakefile,'rb')
        fakedata= pickle.load(fakefile)
        fakefile.close()
        #Calculate distance
        ds, ls, bs, rs, grs, fehs= [], [], [], [], [], []
        for ii in range(len(fakedata)):
            ds.append(_ivezic_dist(fakedata[ii][1],fakedata[ii][0],fakedata[ii][2]))
            ls.append(fakedata[ii][3]+(2*numpy.random.uniform()-1.)\
                          *1.49)
            bs.append(fakedata[ii][4]+(2*numpy.random.uniform()-1.)\
                          *1.49)
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
    if options.fake:
        bovy_plot.bovy_end_print(os.path.join(args[0],options.type+'_'
                                              +'fake_'+
                                              options.sample+'_'+
                                              options.metal+'.'+ext))
    else:
        bovy_plot.bovy_end_print(os.path.join(args[0],options.type+'_'
                                              +options.sample+'_'+
                                              options.metal+'.'+ext))
def afeh(options,args):
    """Plot the [alpha/Fe] vs. [Fe/H] distribution of the sample"""
    if options.png: ext= 'png'
    else: ext= 'ps'
    #Load data
    XYZ,vxvyvz,cov_vxvyvz,data= readData(metal='all',
                                         sample=options.sample)
    bovy_plot.bovy_print()
    bovy_plot.scatterplot(data.feh,data.afe,'k,',
                          xrange=[-2.,0.6],
                          yrange=[-0.1,0.6],
                          xlabel=r'$[\mathrm{Fe/H}]$',
                          ylabel=r'$[\alpha/\mathrm{Fe}]$',
                          onedhists=True)
    #Overplot cuts, metal-rich
    lw=1.3
    bovy_plot.bovy_plot(_APOORFEHRANGE,[_APOORAFERANGE[0],_APOORAFERANGE[0]],
                         'k--',overplot=True,lw=lw)
    bovy_plot.bovy_plot(_APOORFEHRANGE,[_APOORAFERANGE[1],_APOORAFERANGE[1]],
                        'k--',overplot=True,lw=lw)
    bovy_plot.bovy_plot([_APOORFEHRANGE[0],_APOORFEHRANGE[0]],_APOORAFERANGE,
                        'k--',overplot=True,lw=lw)
    bovy_plot.bovy_plot([_APOORFEHRANGE[1],_APOORFEHRANGE[1]],_APOORAFERANGE,
                        'k--',overplot=True,lw=lw)
    #metal-poor
    bovy_plot.bovy_plot(_ARICHFEHRANGE,[_ARICHAFERANGE[0],_ARICHAFERANGE[0]],
                         'k--',overplot=True,lw=lw)
    bovy_plot.bovy_plot(_ARICHFEHRANGE,[_ARICHAFERANGE[1],_ARICHAFERANGE[1]],
                        'k--',overplot=True,lw=lw)
    bovy_plot.bovy_plot([_ARICHFEHRANGE[0],_ARICHFEHRANGE[0]],_ARICHAFERANGE,
                        'k--',overplot=True,lw=lw)
    bovy_plot.bovy_plot([_ARICHFEHRANGE[1],_ARICHFEHRANGE[1]],_ARICHAFERANGE,
                        'k--',overplot=True,lw=lw)
    #metal-richrich
    bovy_plot.bovy_plot([-0.25,-0.25],_APOORAFERANGE,
                        'k:',overplot=True,lw=lw)
    #metal-poorpoor
    bovy_plot.bovy_plot([-0.7,-0.7],_ARICHAFERANGE,
                        'k:',overplot=True,lw=lw)
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
    parser.add_option("-i",dest='fakefile',
                      help="Pickle file with the fake data")
    parser.add_option("--fake",action="store_true", dest="fake",
                      default=False,
                      help="Data is fake")
    return parser


if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    if options.type.lower() == 'datarz' or options.type.lower() == 'dataxy':
        scatterData(options,args)
    elif options.type.lower() == 'afeh':
        afeh(options,args)
    else:
        compareGRichRdist(options,args)
