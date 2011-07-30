PRO ADD_HEALPIX_SEGUEPLATES, infile, outfile
;;add two levels of healpix info in l,b
in= mrdfits(infile,1)
glactc, in.ra, in.dec, 2000., l,b,1, /degree
;;calculate healpix levels for each
ang2pix_ring, 1, !DPI/2.-b/180.*!DPI, l/180.*!DPI, ipring1
ang2pix_ring, 2, !DPI/2.-b/180.*!DPI, l/180.*!DPI, ipring2
xtra= {healpix_level1:0L,healpix_level2:0L}
xtra= replicate(xtra,n_elements(in.ra))
xtra.healpix_level1= ipring1
xtra.healpix_level2= ipring2
out= struct_combine(in,xtra)
mwrfits, out, outfile, /create
END
