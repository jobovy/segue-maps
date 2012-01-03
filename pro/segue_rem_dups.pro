PRO SEGUE_REM_DUPS, infile, outfile
in= mrdfits(infile,1)
spherematch, in.ra, in.dec, in.ra, in.dec, 0.5/3600., m1, m2, maxmatch=0
sortindx= sort(m1)
m1= m1[sortindx]
m2= m2[sortindx]
nmatches= n_elements(m1)
ii=0L
sub= 0L
bestindx= lonarr(n_elements(in.ra))-1
while ii lt nmatches-2 do begin
    print, format = '("Working on ",i7," of ",i7,a1,$)' $
      , (ii+1), nmatches, string(13b)
    if ii eq nmatches-1 then begin
        bestindx[sub]= m2[ii]
        break
    endif
    jj=0
    while (m1[ii] eq m1[ii+1]) and ii lt nmatches-2 do begin
        jj+= 1
        ii+= 1
    endwhile
    if jj gt 0 then begin
        ;;these have multiple matches
        if tag_exist(in,'sna') then begin
            sns= in[m2[ii-jj:ii]].sna
        endif else if tag_exist(in,'snr') then begin
            sns= in[m2[ii-jj:ii]].snr
        endif else if tag_exist(in,'vraderr') then begin
            sns= in[m2[ii-jj:ii]].vraderr
        endif
        maxsn= max(sns,indx)
        bestindx[sub]= m2[ii-jj+indx]
    endif else begin
        bestindx[sub]= m2[ii]
    endelse
    ii+= +1
    sub+= 1
endwhile
;;print, bestindx
bestindx= bestindx[where(bestindx ge 0)]
bestindx= bestindx[uniq(bestindx,sort(bestindx))]
out= in[bestindx]
mwrfits, out, outfile, /create
END
