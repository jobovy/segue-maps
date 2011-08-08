OUTDIR=../figs
EXT=png
NSAMPLES=1000

richdblexp: $(OUTDIR)/HWR_rich_g_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_g_north_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_g_south_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_g_bmin45_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_g_bmax45_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gbright_rfunc.$(EXT) \
	$(OUTDIR)/HWR_rich_gfaint_rfunc.$(EXT)

richflare: $(OUTDIR)/HWR_rich_g_flare_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gbright_flare_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gfaint_flare_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gnorth_flare_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gsouth_flare_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gbmin45_flare_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gbmax45_flare_hRhz.$(EXT)

richtwodblexp: $(OUTDIR)/HWR_rich_g_twoblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gbright_twoblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gfaint_twoblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gnorth_twoblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gsouth_twoblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gbmin45_twoblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gbmax45_twoblexp_hRhz.$(EXT)

richthreedblexp: $(OUTDIR)/HWR_rich_g_threeblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gbright_threeblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gfaint_threeblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gnorth_threeblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gsouth_threeblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gbmin45_threeblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_rich_gbmax45_threeblexp_hRhz.$(EXT)

richtwodblexpflare: $(OUTDIR)/HWR_rich_g_twoblexpflare_hRhz.$(EXT)

poordblexp: $(OUTDIR)/HWR_poor_g_hRhz.$(EXT) \
	$(OUTDIR)/HWR_poor_g_north_hRhz.$(EXT) \
	$(OUTDIR)/HWR_poor_g_south_hRhz.$(EXT) \
	$(OUTDIR)/HWR_poor_g_bmin45_hRhz.$(EXT) \
	$(OUTDIR)/HWR_poor_g_bmax45_hRhz.$(EXT) \
	$(OUTDIR)/HWR_poor_gbright_rfunc.$(EXT) \
	$(OUTDIR)/HWR_poor_gfaint_rfunc.$(EXT)

poorflare: $(OUTDIR)/HWR_poor_g_flare_rfunc.$(EXT) \
	$(OUTDIR)/HWR_poor_gbright_flare_hRhz.$(EXT) \
	$(OUTDIR)/HWR_poor_gfaint_flare_hRhz.$(EXT) \
	$(OUTDIR)/HWR_poor_gnorth_flare_hRhz.$(EXT) \
	$(OUTDIR)/HWR_poor_gsouth_flare_hRhz.$(EXT) \
	$(OUTDIR)/HWR_poor_gbmin45_flare_rfunc.$(EXT) \
	$(OUTDIR)/HWR_poor_gbmax45_flare_rfunc.$(EXT)

poortwodblexp: $(OUTDIR)/HWR_poor_g_twodblexp_rfunc.$(EXT) \
	$(OUTDIR)/HWR_poor_gbright_twodblexp_rfunc.$(EXT) \
	$(OUTDIR)/HWR_poor_gfaint_twodblexp_rfunc.$(EXT) \
	$(OUTDIR)/HWR_poor_gnorth_twodblexp_rfunc.$(EXT) \
	$(OUTDIR)/HWR_poor_gsouth_twodblexp_rfunc.$(EXT) \
	$(OUTDIR)/HWR_poor_gbmin45_twodblexp_rfunc.$(EXT) \
	$(OUTDIR)/HWR_poor_gbmax45_twodblexp_rfunc.$(EXT) 

poorthreedblexp: $(OUTDIR)/HWR_poor_g_threedblexp_rfunc.$(EXT)

poortwoblexpflare: $(OUTDIR)/HWR_poor_g_twoblexpflare_hRhz.$(EXT)

fehranges: $(OUTDIR)/HWR_richpoor_g_twodblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_richrich_g_twodblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_richpoorest_g_twodblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_poorpoor_g_twodblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_poorrich_g_twodblexp_hRhz.$(EXT) \


aferanges: $(OUTDIR)/HWR_apoorpoor_g_twodblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_apoorrich_g_twodblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_arichpoor_g_twodblexp_hRhz.$(EXT) \
	$(OUTDIR)/HWR_arichrich_g_twodblexp_hRhz.$(EXT)

variations: $(OUTDIR)/HWR_rich_g_twodblexp_logg4.2.$(EXT) \
	$(OUTDIR)/HWR_rich_g_twodblexp_minsn30.$(EXT)

#VARIATIONS
$(OUTDIR)/HWR_rich_g_twodblexp_logg4.2.$(EXT): fitDensz.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_logg4.2.sav -o $@ \
	-n $(NSAMPLES) --markovpy --loggmin=4.2 --model=twodblexp
$(OUTDIR)/HWR_rich_g_twodblexp_minsn30.$(EXT): fitDensz.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_minsn30.sav -o $@ \
	-n $(NSAMPLES) --markovpy --snmin=30. --model=twodblexp

#Metal rich G dwarf densities
$(OUTDIR)/HWR_rich_g_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g.sav -o $@ \
	-n $(NSAMPLES) --markovpy
$(OUTDIR)/HWR_rich_g_hRhz_test.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_test.sav -o $@ \
	-n $(NSAMPLES) --markovpy
$(OUTDIR)/HWR_rich_g_mh_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_mh.sav -o $@ \
	-n $(NSAMPLES) --metropolis
$(OUTDIR)/HWR_rich_g_markovpy_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_markovpy.sav -o $@ \
	-n $(NSAMPLES) --markovpy

#Binned color distribution
$(OUTDIR)/HWR_rich_g_colorbinned_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_colorbinned.sav -o $@ \
	-n $(NSAMPLES)  --markovpy --colordist=binned #\

#l,b ranges
$(OUTDIR)/HWR_rich_g_lcen180_bcen90_lbdr30_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_lcen180_bcen90_lbdr30.sav -o $@ \
	-n $(NSAMPLES) --lcen=180. --bcen=90 --lbdr=30 --markovpy
$(OUTDIR)/HWR_rich_g_lcen180_bcen0_lbdr30_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_lcen180_bcen0_lbdr30.sav -o $@ \
	-n $(NSAMPLES) --lcen=180. --bcen=0 --lbdr=30 --markovpy
$(OUTDIR)/HWR_rich_g_lcen45_bcen-23_lbdr20_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_lcen45_bcen-23_lbdr20.sav -o $@ \
	-n $(NSAMPLES) --lcen=45. --bcen=-23 --lbdr=20 --markovpy
$(OUTDIR)/HWR_rich_g_north_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_north.sav -o $@ \
	-n $(NSAMPLES) --north --markovpy
$(OUTDIR)/HWR_rich_g_south_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_south.sav -o $@ \
	-n $(NSAMPLES) --south --markovpy
$(OUTDIR)/HWR_rich_g_bmin45_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_bmin45.sav -o $@ \
	-n $(NSAMPLES) --bmin=45. --markovpy
$(OUTDIR)/HWR_rich_g_bmax45_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_bmax45.sav -o $@ \
	-n $(NSAMPLES) --bmax=45. --markovpy

#Min KS
$(OUTDIR)/HWR_rich_g_minks0.1_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_minks_0.1.sav -o $@ \
	-n $(NSAMPLES) --minks=0.1 --markovpy
$(OUTDIR)/HWR_rich_g_minks-2_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_minks_-2.sav -o $@ \
	-n $(NSAMPLES) --minks=-2 --markovpy
#Min plateSN_r
$(OUTDIR)/HWR_rich_g_minplatesn70_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_minplatesn_70.sav -o $@ \
	-n $(NSAMPLES) --minplatesn=70 --markovpy
$(OUTDIR)/HWR_rich_g_minplatesn85_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_minplatesn_85.sav -o $@ \
	-n $(NSAMPLES) --minplatesn=85 --markovpy
$(OUTDIR)/HWR_rich_g_minplatesn70_maxplatesn85hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_minplatesn_70_maxplatesn85.sav -o $@ \
	-n $(NSAMPLES) --minplatesn_faint=70 --maxplatesn_faint=85 \
	--minplatesn_bright=0. --markovpy
$(OUTDIR)/HWR_rich_g_minplatesn10_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_minplatesn_10.sav -o $@ \
	-n $(NSAMPLES) --minplatesn=10 --markovpy

#TwoVertical model
$(OUTDIR)/HWR_rich_g_twovertical_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_twovertical.sav -o $@ \
	-n $(NSAMPLES) --model=twovertical --markovpy
#TwoDblExp model
$(OUTDIR)/HWR_rich_g_twoblexp_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --model=twodblexp --markovpy
$(OUTDIR)/HWR_rich_gbright_twoblexp_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gbright_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --model=twodblexp --markovpy --bright
$(OUTDIR)/HWR_rich_gfaint_twoblexp_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gfaint_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --model=twodblexp --markovpy --faint
$(OUTDIR)/HWR_rich_gnorth_twoblexp_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gnorth_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --model=twodblexp --markovpy --north
$(OUTDIR)/HWR_rich_gsouth_twoblexp_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gsouth_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --model=twodblexp --markovpy --south
$(OUTDIR)/HWR_rich_gbmin45_twoblexp_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gbmin45_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --model=twodblexp --markovpy --bmin=45.
$(OUTDIR)/HWR_rich_gbmax45_twoblexp_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gbmax45_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --model=twodblexp --markovpy --bmax=45.
#ThreeDblExp model
$(OUTDIR)/HWR_rich_g_threeblexp_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_threedblexp.sav -o $@ \
	-n $(NSAMPLES) --model=threedblexp --markovpy
$(OUTDIR)/HWR_rich_gbright_threeblexp_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gbright_threedblexp.sav -o $@ \
	-n $(NSAMPLES) --model=threedblexp --markovpy --bright
$(OUTDIR)/HWR_rich_gfaint_threeblexp_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gfaint_threedblexp.sav -o $@ \
	-n $(NSAMPLES) --model=threedblexp --markovpy --faint
$(OUTDIR)/HWR_rich_gnorth_threeblexp_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gnorth_threedblexp.sav -o $@ \
	-n $(NSAMPLES) --model=threedblexp --markovpy --north
$(OUTDIR)/HWR_rich_gsouth_threeblexp_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gsouth_threedblexp.sav -o $@ \
	-n $(NSAMPLES) --model=threedblexp --markovpy --south
$(OUTDIR)/HWR_rich_gbmin45_threeblexp_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gbmin45_threedblexp.sav -o $@ \
	-n $(NSAMPLES) --model=threedblexp --markovpy --bmin=45.
$(OUTDIR)/HWR_rich_gbmax45_threeblexp_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gbmax45_threedblexp.sav -o $@ \
	-n $(NSAMPLES) --model=threedblexp --markovpy --bmax=45.
#TwoDblExpFlare model
$(OUTDIR)/HWR_rich_g_twoblexpflare_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_twodblexpflare.sav -o $@ \
	-n $(NSAMPLES) --model=twodblexpflare --markovpy
#Flare model
$(OUTDIR)/HWR_rich_g_flare_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_flare.sav -o $@ \
	-n $(NSAMPLES) --model=flare --markovpy
$(OUTDIR)/HWR_rich_gbright_flare_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gbright_flare.sav -o $@ \
	-n $(NSAMPLES) --model=flare --markovpy --bright
$(OUTDIR)/HWR_rich_gfaint_flare_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gfaint_flare.sav -o $@ \
	-n $(NSAMPLES) --model=flare --markovpy --faint
$(OUTDIR)/HWR_rich_gnorth_flare_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gnorth_flare.sav -o $@ \
	-n $(NSAMPLES) --model=flare --markovpy --north
$(OUTDIR)/HWR_rich_gsouth_flare_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gsouth_flare.sav -o $@ \
	-n $(NSAMPLES) --model=flare --markovpy --south
$(OUTDIR)/HWR_rich_gbmin45_flare_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gbmin45_flare.sav -o $@ \
	-n $(NSAMPLES) --model=flare --markovpy --bmin=45.
$(OUTDIR)/HWR_rich_gbmax45_flare_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gbmax45_flare.sav -o $@ \
	-n $(NSAMPLES) --model=flare --markovpy --bmax=45.
#Min KS
$(OUTDIR)/HWR_rich_g_flare_minks0.1_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_flare_minks_0.1.sav -o $@ \
	-n $(NSAMPLES) --minks=0.1 --model=flare --markovpy
#Tied Flare model
$(OUTDIR)/HWR_rich_g_tiedflare_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_g_tiedflare.sav -o $@ \
	-n $(NSAMPLES) --model=tiedflare --markovpy

#just bright
$(OUTDIR)/HWR_rich_gbright_rfunc.$(EXT): fitDensz.py Makefile plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gbright.sav -o $@ \
	-n $(NSAMPLES) --bright --markovpy
$(OUTDIR)/TwoVertical_rich_gbright_rfunc.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py TwoVertical_dens_rich_gbright.sav -o $@ \
	-n $(NSAMPLES) --bright --model=twovertical --markovpy
$(OUTDIR)/Flare_rich_gbright_rfunc.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py Flare_dens_rich_gbright.sav -o $@ \
	-n $(NSAMPLES) --bright --model=flare --markovpy
$(OUTDIR)/DblExp_rich_gbright_rfunc.$(EXT): fitDensz.py Makefile plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py DblExp_dens_rich_gbright.sav -o $@ \
	-n $(NSAMPLES) --bright --model=dblexp --markovpy
#just faint
$(OUTDIR)/HWR_rich_gfaint_rfunc.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_gfaint.sav -o $@ \
	-n $(NSAMPLES) --faint --markovpy
$(OUTDIR)/TwoVertical_rich_gfaint_rfunc.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py TwoVertical_dens_rich_gfaint.sav -o $@ \
	-n $(NSAMPLES) --faint --model=twovertical --markovpy
$(OUTDIR)/Flare_rich_gfaint_rfunc.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py Flare_dens_rich_gfaint.sav -o $@ \
	-n $(NSAMPLES) --faint --model=flare --markovpy
$(OUTDIR)/DblExp_rich_gfaint_rfunc.$(EXT): fitDensz.py Makefile plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py DblExp_dens_rich_gfaint.sav -o $@ \
	-n $(NSAMPLES) --faint --model=dblexp --markovpy

#Fake data
$(OUTDIR)/HWR_rich_fakedblexp_g_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_fakedblexp_g.sav -o $@ \
	-n $(NSAMPLES) --sel_faint=r --fake -i fakeThinDblExp.sav \
	 --markovpy
$(OUTDIR)/HWR_rich_fakedblexp_gbright_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_fakedblexp_gbright.sav -o $@ \
	-n $(NSAMPLES) --sel_faint=r --fake -i fakeThinDblExp.sav \
	 --markovpy
$(OUTDIR)/Flare_rich_fakedblexp_g_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py Flare_dens_rich_fakedblexp_g.sav -o $@ \
	-n $(NSAMPLES) --sel_faint=r --fake -i fakeThinDblExp.sav \
	--model=flare --markovpy
$(OUTDIR)/Flare_rich_fakeflare_g_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py Flare_dens_rich_fakedblexp_g.sav -o $@ \
	-n $(NSAMPLES) --sel_faint=r --fake -i fakeThinFlare.sav \
	--model=flare --markovpy
$(OUTDIR)/HWR_rich_fakeflare_g_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_rich_fakeflare_g.sav -o $@ \
	-n $(NSAMPLES) --sel_faint=r --fake -i fakeThinFlare.sav \
	 --markovpy

#Metal poor G dwarf densities
$(OUTDIR)/HWR_poor_g_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g.sav -o $@ \
	-n $(NSAMPLES) --metal=poor --markovpy
#l,b ranges
$(OUTDIR)/HWR_poor_g_lcen180_bcen90_lbdr30_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_lcen180_bcen90_lbdr30.sav -o $@ \
	-n $(NSAMPLES) --lcen=180. --bcen=90 --lbdr=30 --metal=poor \
	 --markovpy
$(OUTDIR)/HWR_poor_g_lcen180_bcen0_lbdr30_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_lcen180_bcen0_lbdr30.sav -o $@ \
	-n $(NSAMPLES) --lcen=180. --bcen=0 --lbdr=30 --metal=poor \
	 --markovpy
$(OUTDIR)/HWR_poor_g_lcen45_bcen-23_lbdr20_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_lcen45_bcen-23_lbdr20.sav -o $@ \
	-n $(NSAMPLES) --lcen=45. --bcen=-23 --lbdr=20 --metal=poor \
	 --markovpy
$(OUTDIR)/HWR_poor_g_north_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_north.sav -o $@ \
	-n $(NSAMPLES) --north --metal=poor \
	 --markovpy
$(OUTDIR)/HWR_poor_g_south_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_south.sav -o $@ \
	-n $(NSAMPLES) --south --metal=poor \
	 --markovpy
$(OUTDIR)/HWR_poor_g_bmin45_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_bmin45.sav -o $@ \
	-n $(NSAMPLES) --bmin=45. --markovpy --metal=poor
$(OUTDIR)/HWR_poor_g_bmax45_hRhz.$(EXT): fitDensz.py Makefile \
	segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_bmax45.sav -o $@ \
	-n $(NSAMPLES) --bmax=45. --markovpy --metal=poor

#min KS
$(OUTDIR)/HWR_poor_g_minks0.1_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_minks_0.1.sav -o $@ \
	-n $(NSAMPLES) --minks=0.1 --metal=poor \
	 --markovpy
$(OUTDIR)/HWR_poor_g_minks0_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_minks_0.sav -o $@ \
	-n $(NSAMPLES) --minks=0 --metal=poor --markovpy
#TwoVertical model
$(OUTDIR)/HWR_poor_g_twovertical_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_twovertical.sav -o $@ \
	-n $(NSAMPLES) --metal=poor \
	--model=twovertical --markovpy
#TwoDblExp model
$(OUTDIR)/HWR_poor_g_twodblexp_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --metal=poor \
	--model=twodblexp --markovpy
$(OUTDIR)/HWR_poor_gbright_twodblexp_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_gbright_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --metal=poor \
	--model=twodblexp --markovpy --bright
$(OUTDIR)/HWR_poor_gfaint_twodblexp_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_gfaint_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --metal=poor \
	--model=twodblexp --markovpy --faint
$(OUTDIR)/HWR_poor_gnorth_twodblexp_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_gnorth_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --metal=poor \
	--model=twodblexp --markovpy --north
$(OUTDIR)/HWR_poor_gsouth_twodblexp_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_gsouth_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --metal=poor \
	--model=twodblexp --markovpy --south
$(OUTDIR)/HWR_poor_gbmin45_twodblexp_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_gbmin45_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --metal=poor \
	--model=twodblexp --markovpy --bmin=45.
$(OUTDIR)/HWR_poor_gbmax45_twodblexp_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_gbmax45_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --metal=poor \
	--model=twodblexp --markovpy --bmax=45.
#Threedblexp
$(OUTDIR)/HWR_poor_g_threedblexp_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_threedblexp.sav -o $@ \
	-n $(NSAMPLES) --metal=poor \
	--model=threedblexp --markovpy
#TwoDblExpFlare model
$(OUTDIR)/HWR_poor_g_twoblexpflare_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_twodblexpflare.sav -o $@ \
	-n $(NSAMPLES) --model=twodblexpflare --markovpy --metal=poor
#Flare model
$(OUTDIR)/HWR_poor_g_flare_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_flare.sav -o $@ \
	-n $(NSAMPLES) --metal=poor \
	--model=flare --markovpy
$(OUTDIR)/HWR_poor_gbright_flare_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_gbright_flare.sav -o $@ \
	-n $(NSAMPLES) --metal=poor \
	--model=flare --markovpy --bright
$(OUTDIR)/HWR_poor_gfaint_flare_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_gfaint_flare.sav -o $@ \
	-n $(NSAMPLES) --metal=poor \
	--model=flare --markovpy --faint
$(OUTDIR)/HWR_poor_gnorth_flare_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_gnorth_flare.sav -o $@ \
	-n $(NSAMPLES) --metal=poor \
	--model=flare --markovpy --north
$(OUTDIR)/HWR_poor_gsouth_flare_hRhz.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_gsouth_flare.sav -o $@ \
	-n $(NSAMPLES) --metal=poor \
	--model=flare --markovpy --south
$(OUTDIR)/HWR_poor_gbmin45_flare_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_gbmin45_flare.sav -o $@ \
	-n $(NSAMPLES) --metal=poor \
	--model=flare --markovpy --bmin=45.
$(OUTDIR)/HWR_poor_gbmax45_flare_rfunc.$(EXT): fitDensz.py Makefile \
	plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_gbmax45_flare.sav -o $@ \
	-n $(NSAMPLES) --metal=poor \
	--model=flare --markovpy --bmax=45.
#Min KS
$(OUTDIR)/HWR_poor_g_flare_minks0.1_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_g_flare_minks_0.1.sav -o $@ \
	-n $(NSAMPLES) --minks=0.1 --model=flare --metal=poor --markovpy
#just bright
$(OUTDIR)/HWR_poor_gbright_rfunc.$(EXT): fitDensz.py Makefile plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_gbright.sav -o $@ \
	-n $(NSAMPLES) --bright --metal=poor --markovpy
#just bright
$(OUTDIR)/HWR_poor_gfaint_rfunc.$(EXT): fitDensz.py Makefile plotData.py segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poor_gfaint.sav -o $@ \
	-n $(NSAMPLES) --faint --metal=poor --markovpy

#Metal rich G dwarf densities, flare
$(OUTDIR)/flare_rich_g_hRhz.$(EXT): fitDensz.py Makefile
	$(PYTHON) fitDensz.py flare_dens_rich_g.sav -o $@ \
	-n $(NSAMPLES) --model=flare --markovpy

#Metal poor G dwarf densities, flare
$(OUTDIR)/flare_poor_g_hRhz.$(EXT): fitDensz.py Makefile
	$(PYTHON) fitDensz.py flare_dens_poor_g.sav -o $@ \
	-n $(NSAMPLES) --metal=poor --model=flare --markovpy

#Poor poor
$(OUTDIR)/HWR_poorpoor_g_twodblexp_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poorpoor_g_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --metal=poorpoor --markovpy --model=twodblexp
#Poor rich
$(OUTDIR)/HWR_poorrich_g_twodblexp_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_poorrich_g_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --metal=poorrich --markovpy --model=twodblexp
#Rich poor
$(OUTDIR)/HWR_richpoor_g_twodblexp_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_richpoor_g_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --metal=richpoor --markovpy --model=twodblexp
#Rich rich
$(OUTDIR)/HWR_richrich_g_twodblexp_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_richrich_g_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --metal=richrich --markovpy --model=twodblexp
#Rich poorest
$(OUTDIR)/HWR_richpoorest_g_twodblexp_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_richpoorest_g_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --metal=richpoorest --markovpy --model=twodblexp


#Poor poor
$(OUTDIR)/HWR_apoorpoor_g_twodblexp_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_apoorpoor_g_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --metal=apoorpoor --markovpy --model=twodblexp
#Poor rich
$(OUTDIR)/HWR_apoorrich_g_twodblexp_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_apoorrich_g_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --metal=apoorrich --markovpy --model=twodblexp
#Rich poor
$(OUTDIR)/HWR_arichpoor_g_twodblexp_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_arichpoor_g_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --metal=arichpoor --markovpy --model=twodblexp
#Rich rich
$(OUTDIR)/HWR_arichrich_g_twodblexp_hRhz.$(EXT): fitDensz.py Makefile segueSelect.py
	$(PYTHON) fitDensz.py HWR_dens_arichrich_g_twodblexp.sav -o $@ \
	-n $(NSAMPLES) --metal=arichrich --markovpy --model=twodblexp


