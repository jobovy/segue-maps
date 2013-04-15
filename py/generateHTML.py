import os, os.path
from optparse import OptionParser
def generateHTML(options,args):
    outfile= open(options.outfilename,'w')
    outfile.write('<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n')
    outfile.write('<html xmlns="http://www.w3.org/1999/xhtml">\n')
    outfile.write('<head>\n')
    outfile.write('<meta http-equiv="content-type" content="text/html; charset=utf-8" />\n')
    outfile.write('<title>Fit results for %s dwarfs</title>\n' % (options.sample.upper()))
    outfile.write('<meta name="description" content="Fit results for %s dwarfs" />\n' % (options.sample.upper()))
    outfile.write('</head>')
    outfile.write('<body>')
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    if False:
        types= ['rdfh','rdfhrdvt','rdhr','rdsz',
                'rdpout','rddvt','srsz','pout','dvt',
                'loglhist','props','rdfhrdf','rdfhrdvtrdf']
    types= ['rdfh','rdhr','rdhrc','rdsz',
            'rdpout','rddvt','srsz','srszc','pout',
            'loglhist','props']
    ntypes= len(types)
    for ii in range(npops):
        outfile.write('<p style="font-size:xx-large;">%i</p>\n' % ii)
        line= ''
        for jj in range(ntypes):
            line+= '<img src="%s" alt="" width="300" /> ' % (os.path.join(options.figdir,options.basename+'_%s_%i.png' % (types[jj],ii)))
        outfile.write(line+'<br>\n')
        outfile.write('<hr size="3" color="black">\n')
    outfile.write('</body>\n')
    outfile.write('</html>\n')
    outfile.close()

def get_options():
    usage = "usage: %prog [options] <savefile>\n\nsavefile= name of the file that the fits will be saved to"
    parser = OptionParser(usage=usage)
    #Data options
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--figdir",dest='figdir',default=None,
                      help="relative path of the figure directory")
    parser.add_option("--basename",dest='basename',default=None,
                      help="Base of the name of each file")    
    parser.add_option("-o","--outfilename",dest='outfilename',default=None,
                      help="Name for an output file")
    return parser

if __name__  == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    generateHTML(options,args)
