OUTDIR=../figs
EXT=png

$(OUTDIR)/HWR_rich_g_sR.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_rich_g.sav -o $@ --expd1 --expd2 -n 10000 \
	--xmin=15. --xmax=25. --ymin=0. --ymax=30. \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='R_\sigma\ [\mathrm{kpc}]'

$(OUTDIR)/HWR_rich_g_spbad.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_rich_g.sav -o $@ --expd1 -n 10000 \
	--xmin=15. --xmax=25. --ymin=0. --ymax=.05 --d2=0 \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='P_{\mathrm{bad}}'