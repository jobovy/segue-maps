import os, os.path
import numpy
from optparse import OptionParser
from galpy.util import bovy_plot
import segueSelect
from fitSigz import readData
from fitDensz import _HWRDensity, _FlareDensity, _const_colordist
import compareDataModel
def compareGRichRdist(options,args):
    #Set up density models and their parameters
    model1= _FlareDensity
    model2= _HWRDensity
    params1=  numpy.array([-1.20815403035,0.981614534466,-0.130775138707])
    params2= numpy.log(numpy.array([0.3,2.45,1.]))
    #Load sf
    sf= segueSelect.segueSelect(sample=options.sample,sn=True,
                                type_bright='constant',
                                type_faint='r')
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
            bovy_plot.bovy_end_print(os.path.join(args[0],'Flare_Dblexp_g_rich_l%i_b%i_bright.ps' % (ls[ii],bs[ii])))
        else:
            bovy_plot.bovy_end_print(os.path.join(args[0],'Flare_Dblexp_g_rich_zdist_l%i_b%i_bright.ps' % (ls[ii],bs[ii])))
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
            bovy_plot.bovy_end_print(os.path.join(args[0],'Flare_Dblexp_g_rich_l%i_b%i_faint.ps' % (ls[ii],bs[ii])))
        elif options.type == 'z':
            bovy_plot.bovy_end_print(os.path.join(args[0],'Flare_Dblexp_g_rich_zdist_l%i_b%i_faint.ps' % (ls[ii],bs[ii])))
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
    return parser


if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    if options.sample.lower() == 'g' and options.metal.lower() == 'rich':
        compareGRichRdist(options,args)
