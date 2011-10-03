;+
;   NAME:
;      add_ysl_afe
;   PURPOSE:
;      add the DR8 afe, feh, logg, and teff from Young Sun Lee
;   INPUT:
;      infile - old file
;   OUTPUT:
;      outfile - new file
;   HISTORY:
;      2011-02-03 - Written - Bovy (IAS)
;-
PRO ADD_YSL_AFE, infile, outfile, kdwarf= kdwarf
old= mrdfits(infile,1,/silent)
;;First save the old values
dr7str= {feh_dr7: 0D, feh_err_dr7: 0D, $
         afe_dr7: 0D, afe_err_dr7: 0D, $
         logga_dr7: 0D, logg_err_dr7: 0D, $
         teff_dr7: 0D, teff_err_dr7: 0D}
dr7str= replicate(dr7str,n_elements(old))
dr7str.feh_dr7= old.feh
dr7str.feh_err_dr7= old.feh_err
dr7str.afe_dr7= old.afe
dr7str.afe_err_dr7= old.afe_err
dr7str.teff_dr7= old.teff
dr7str.teff_err_dr7= old.tef_err
dr7str.logga_dr7= old.logga
dr7str.logg_err_dr7= old.logg_err
old= struct_combine(old,dr7str)
;;Now load the YSL files and match
IF keyword_set(kdwarf) THEN $
  new_files= file_search('$DATADIR/bovy/segue-local/kdwarfs/ssppOut*') $
ELSE $
  new_files= file_search('$DATADIR/bovy/segue-local/gdwarfs/ssppOut*')
nfiles= n_elements(new_files)
;;create index to keep track of whether everything gets matched
matched= bytarr(n_elements(old))
for ii=0L, nfiles-1 do begin
    new= mrdfits(new_files[ii],1,/silent)
    spherematch, old.ra, old.dec, new.ra, new.dec, 2./3600., oindx, nindx
    old[oindx].feh= new[nindx].feh_adop
    old[oindx].feh_err= new[nindx].feh_adop_unc
    old[oindx].afe= new[nindx].afe
    old[oindx].afe_err= new[nindx].afe_unc
    old[oindx].teff= new[nindx].teff_adop
    old[oindx].tef_err= new[nindx].teff_adop_unc
    old[oindx].logga= new[nindx].logg_adop
    old[oindx].logg_err= new[nindx].logg_adop_unc
    ;;matched
    matched[oindx]= 1
endfor
dum= where(matched EQ 1,nmatched)
if nmatched NE n_elements(old) then begin
    print, "Warning: only "+strtrim(string(nmatched),2)+" out of "+$
      strtrim(string(n_elements(old)),2)+" got matched"
endif
;;write output
mwrfits, old, outfile, /create
END
