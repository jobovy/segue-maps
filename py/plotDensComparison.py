import os, os.path
import numpy
from optparse import OptionParser
from galpy.util import bovy_plot
import segueSelect
from fitSigz import readData
from fitDensz import _HWRDensity, _FlareDensity, _const_colordist
import compareDataModel
def compareGRichRdist(options,args):
    if options.png: ext= 'png'
    else: ext= 'ps'
    #Set up density models and their parameters
    model1= _FlareDensity
    model2= _HWRDensity
    if options.metal.lower() == 'rich':
        params1=  numpy.array([-1.20172829533,1.01068814092,-0.0464210825653])
        params2= numpy.array([-1.10521544525,0.945523259073,0.00824201794418])
    else:
        #params1=  numpy.array([0.187391923558,0.71285154528,1.30084421599])
        #params2= numpy.array([0.108148171668,2.46676108062,0.00206572947631])
        params1=  numpy.array([-0.187391923558,0.71285154528,1.30084421599])
        params2= numpy.array([-0.108148171668,2.46676108062,0.00206572947631])
    #Load sf
    sf= segueSelect.segueSelect(sample=options.sample,sn=True,
                                type_bright='sharprcut',
                                type_faint='sharprcut')
    #Load data
    XYZ,vxvyvz,cov_vxvyvz,data= readData(metal=options.metal,
                                         sample=options.sample)
    #We do bright/faint for 4 directions
    ls= [180,180,45,45]
    bs= [0,90,-23,23]
    bins= 21
    #Set up comparison
    if options.type == 'r':
        compare_func= compareDataModel.comparerdistPlate
    elif options.type == 'z':
        compare_func= compareDataModel.comparezdistPlate
    for ii in range(len(ls)):
        #Bright
        plate= compareDataModel.similarPlatesDirection(ls[ii],bs[ii],20.,
                                                       sf,data,
                                                       faint=False)
        bovy_plot.bovy_print()
        compare_func(model1,params1,sf,_const_colordist,
                     data,plate,color='k',
                     rmin=14.5,rmax=20.2,grmin=0.48,grmax=0.55,
                     bins=bins,ls='-')
        compare_func(model2,params2,sf,_const_colordist,
                     data,plate,color='k',bins=bins,
                     rmin=14.5,rmax=20.2,grmin=0.48,grmax=0.55,
                     overplot=True,ls='--')
        if options.type == 'r':
            bovy_plot.bovy_end_print(os.path.join(args[0],'Flare_Dblexp_g_'+options.metal+'_l%i_b%i_bright.' % (ls[ii],bs[ii]))+ext)
        else:
            bovy_plot.bovy_end_print(os.path.join(args[0],'Flare_Dblexp_g_'+options.metal+'_zdist_l%i_b%i_bright.' % (ls[ii],bs[ii]))+ext)
        #Faint
        plate= compareDataModel.similarPlatesDirection(ls[ii],bs[ii],20.,
                                                       sf,data,
                                                       bright=False)
        bovy_plot.bovy_print()
        compare_func(model1,params1,sf,_const_colordist,
                     data,plate,color='k',
                     rmin=14.5,rmax=20.2,grmin=0.48,grmax=0.55,
                     bins=bins,ls='-')
        compare_func(model2,params2,sf,_const_colordist,
                     data,plate,color='k',bins=bins,
                     rmin=14.5,rmax=20.2,grmin=0.48,grmax=0.55,
                     overplot=True,ls='--')
        if options.type == 'r':
            bovy_plot.bovy_end_print(os.path.join(args[0],'Flare_Dblexp_g_'+options.metal+'_l%i_b%i_faint.' % (ls[ii],bs[ii]))+ext)
        elif options.type == 'z':
            bovy_plot.bovy_end_print(os.path.join(args[0],'Flare_Dblexp_g_'+options.metal+'_zdist_l%i_b%i_faint.' % (ls[ii],bs[ii]))+ext)
    return None

def get_options():
    usage = "usage: %prog [options] <savedir>\n\nsavedir= name of the directory that the comparisons will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--metal",dest='metal',default='rich',
                      help="Use metal-poor or rich sample ('poor', 'rich' or 'all')")
    parser.add_option("-t","--type",dest='type',default='r',
                      help="Type of comparison to make ('r', 'z')")
    parser.add_option("--png",action="store_true", dest="png",
                      default=False,
                      help="Save as png, otherwise ps")
    return parser


if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    compareGRichRdist(options,args)
