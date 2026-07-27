/* ============================================================
   Worked Example: Fixed Effects, Random Effects, and REWB
   Bell, Fairbrother & Jones (2019) — Quality & Quantity 53:1051

   Dataset: Cornwell & Rupert (1988) PSID Males Wages Panel
     545 male household heads, 8 years (1980–1987), N = 4,360
     Sourced from: plm R package (Males dataset)
     Citation: Vella & Verbeek (1998), J. Applied Econometrics

   This .do file reads males_wages.csv (saved by rewb_males_wages.R).
   Both files should be in the same directory.

   Alternative: if you have Stata and internet access, replace the
   import block below with:
       webuse nlswork, clear
   and rename: ln_wage→wage, ttl_exp→exper, grade→school,
               union→union_d  (already 0/1 in nlswork)
   The substantive story (experience and wages) is identical.

   Research question:
     How does work experience relate to log wages —
     within persons (year-by-year growth) vs between persons
     (do experienced workers earn more on average)?

   Models fitted:
     1. Pooled OLS
     2. Fixed Effects (FE)              -- xtreg, fe
     3. Standard Random Effects (RE)    -- xtreg, re
     4. Hausman-style test
     5. Within-Between RE (REWB)        -- mixed
     6. Mundlak parameterisation        -- xtreg, re + group mean
     7. REWB with Random Slopes         -- mixed + random slope

   Replication resources:
     Bell & Jones (2015) .do files: https://doi.org/10.7910/DVN/23415
     Jordan & Philips (2023) qdmean: https://github.com/andyphilips/qdmean

   Stata version: 15+  (mixed command)
   Earlier versions: replace -mixed- with -xtmixed-
============================================================ */

clear all
set more off
version 15

/* ── 1. Load data ────────────────────────────────────────────
   males_wages.csv must be in the working directory.
   To check/set: pwd  and  cd "path/to/folder"
────────────────────────────────────────────────────────── */
import delimited using "males_wages.csv", clear

/* Alternatively, adjust the full path:
   import delimited using "/path/to/males_wages.csv", clear  */

label variable wage      "Log wage"
label variable exper     "Work experience (years)"
label variable school    "Years of schooling (time-invariant)"
label variable union_d   "Union member (1=yes)"
label variable married_d "Married (1=yes)"
label variable health_d  "Poor health (1=yes)"
label variable black_d   "Black (time-invariant)"
label variable hisp_d    "Hispanic (time-invariant)"

xtset id year

di _n "Dataset: PSID Males Wages (Cornwell-Rupert 1988 / plm::Males)"
di    "Individuals: 545  |  Years: 1980-1987  |  N = " _N

/* ── 2. Decompose experience ──────────────────────────────────
   exper_bar: person-specific mean experience (between component)
   exper_win: deviation from own mean         (within component)
────────────────────────────────────────────────────────── */
bysort id: egen exper_bar = mean(exper)
gen  exper_win = exper - exper_bar

label variable exper_bar "Mean experience (between component)"
label variable exper_win "exper minus person mean (within component)"

/* Confirm school is strictly time-invariant */
bysort id: egen school_sd = sd(school)
sum school_sd, meanonly
di _n "Within-person SD of school (should be 0): " r(mean)
drop school_sd

di _n "-----------------------------------------------------------------"
di    "Variable roles:"
di    "  wage      : log wage (outcome)"
di    "  exper     : work experience, years (time-varying)"
di    "  exper_bar : person mean of experience   -- BETWEEN component"
di    "  exper_win : exper minus person mean     -- WITHIN component"
di    "  school    : years of schooling (time-INVARIANT)"
di    "  black_d / hisp_d: ethnicity (time-INVARIANT)"
di    "  union_d / married_d / health_d: time-varying controls"

/* ── 3. MODEL 1: Pooled OLS ──────────────────────────────────
   Coefficient on exper is an uninterpretable blend of within
   (person grows in experience) and between (experienced workers
   differ from inexperienced in many unobserved ways).
   SEs also underestimated because clustering is ignored.
────────────────────────────────────────────────────────── */
di _n "-----------------------------------------------------------------"
di    "MODEL 1: Pooled OLS"
di    "exper coefficient blends within and between effects"
di    "-----------------------------------------------------------------"
regress wage exper school union_d married_d health_d black_d hisp_d

/* ── 4. MODEL 2: Fixed Effects (FE) ──────────────────────────
   Demeaning removes all between-person variation; unbiased
   within estimate.  school, black_d, hisp_d dropped as
   time-invariant — FE cannot say anything about them.
────────────────────────────────────────────────────────── */
di _n "-----------------------------------------------------------------"
di    "MODEL 2: Fixed Effects (FE)"
di    "school / black_d / hisp_d dropped (time-invariant)"
di    "-----------------------------------------------------------------"
xtreg wage exper union_d married_d health_d, fe

/* ── 5. MODEL 3: Standard Random Effects (RE) ────────────────
   Assumes within effect = between effect for exper.
   Hausman test below will show this assumption is rejected.
────────────────────────────────────────────────────────── */
di _n "-----------------------------------------------------------------"
di    "MODEL 3: Standard Random Effects (RE)"
di    "Assumes beta_W = beta_B for exper (check with Hausman test)"
di    "-----------------------------------------------------------------"
xtreg wage exper school union_d married_d health_d black_d hisp_d, re

/* ── 6. Hausman test ──────────────────────────────────────────
   Tests H0: within effect = between effect of exper.
   Significant => effects differ => use REWB, not standard RE.
   NB: this is a test of effect equality, NOT a model selector.
────────────────────────────────────────────────────────── */
di _n "-----------------------------------------------------------------"
di    "Hausman test: H0 = within effect of exper = between effect"
di    "Significant => use REWB"
di    "-----------------------------------------------------------------"
quietly xtreg wage exper union_d married_d health_d, fe
estimates store fe_stored
quietly xtreg wage exper school union_d married_d health_d black_d hisp_d, re
hausman fe_stored ., sigmamore

/* ── 7. MODEL 4: REWB — manual demeaning  (recommended) ──────
   The key model.  Separates within and between effects:
     exper_win → within effect (≈ FE estimate via GLS)
     exper_bar → between effect (new information vs FE)
     school / ethnicity now estimable (unlike FE)
────────────────────────────────────────────────────────── */
di _n "-----------------------------------------------------------------"
di    "MODEL 4: Within-Between RE (REWB)  -- mixed"
di    "exper_win = within effect   (≈ FE estimate)"
di    "exper_bar = between effect  (new vs FE)"
di    "school / black_d / hisp_d  (not estimable by FE)"
di    "-----------------------------------------------------------------"
mixed wage exper_win exper_bar school union_d married_d health_d ///
      black_d hisp_d || id:, reml

/* ── 8. MODEL 5: Mundlak parameterisation ────────────────────
   Raw exper + group mean exper_bar:
     exper     → within effect   (beta_W) — same as REWB exper_win
     exper_bar → contextual eff  (beta_2C, net of own exper)
   Identity: beta_W + beta_2C = beta_2B  (REWB between effect)
   Models are mathematically equivalent; same log-likelihood.
────────────────────────────────────────────────────────── */
di _n "-----------------------------------------------------------------"
di    "MODEL 5: Mundlak parameterisation  -- xtreg, re"
di    "exper     → within effect (beta_W)"
di    "exper_bar → contextual effect (beta_2C)"
di    "Check: beta_W + beta_2C = beta_2B (REWB between)"
di    "-----------------------------------------------------------------"
xtreg wage exper exper_bar school union_d married_d health_d black_d hisp_d, re
estimates store mundlak_stored

/* ── 9. MODEL 6: REWB with Random Slopes ─────────────────────
   Allows the within-experience effect to vary across people.
   Bell et al. (2019, Fig. 1): omitting random slopes when they
   exist produces anti-conservative standard errors in both FE
   and standard RE models.
────────────────────────────────────────────────────────── */
di _n "-----------------------------------------------------------------"
di    "MODEL 6: REWB with Random Slopes  -- mixed"
di    "v_i0 = random intercept; v_i1 = random slope on exper_win"
di    "-----------------------------------------------------------------"
mixed wage exper_win exper_bar school union_d married_d health_d ///
      black_d hisp_d || id: exper_win, reml covariance(unstructured)

/* ── 10. Verification: Mundlak identity ─────────────────────
   beta_W + beta_2C = beta_2B
────────────────────────────────────────────────────────── */
di _n "-----------------------------------------------------------------"
di    "Verification: Mundlak identity  beta_W + beta_2C = beta_2B"
di    "-----------------------------------------------------------------"

quietly xtreg wage exper exper_bar school union_d married_d health_d ///
        black_d hisp_d, re
scalar bW  = _b[exper]
scalar b2C = _b[exper_bar]

quietly mixed wage exper_win exper_bar school union_d married_d health_d ///
             black_d hisp_d || id:, reml
scalar b2B_rewb = _b[exper_bar]

di sprintf("  beta_W  (Mundlak, raw exper):   %8.5f", bW)
di sprintf("  beta_2C (Mundlak, exper_bar):   %8.5f", b2C)
di sprintf("  Sum:                            %8.5f", bW + b2C)
di sprintf("  beta_2B (REWB,    exper_bar):   %8.5f  <-- should match", b2B_rewb)

/* ── 11. Summary Table ───────────────────────────────────────
   Collect key coefficients across models.
────────────────────────────────────────────────────────── */
di _n "================================================================="
di    "SUMMARY: Coefficient on experience across models"
di    "================================================================="

quietly regress wage exper school union_d married_d health_d black_d hisp_d
scalar ols_w = _b[exper]; scalar ols_s = _b[school]; scalar ols_b = _b[black_d]

quietly xtreg wage exper union_d married_d health_d, fe
scalar fe_w  = _b[exper]

quietly xtreg wage exper school union_d married_d health_d black_d hisp_d, re
scalar re_w  = _b[exper]; scalar re_s = _b[school]; scalar re_b = _b[black_d]

quietly mixed wage exper_win exper_bar school union_d married_d health_d ///
             black_d hisp_d || id:, reml
scalar rewb_w = _b[exper_win]; scalar rewb_b2B = _b[exper_bar]
scalar rewb_s = _b[school]; scalar rewb_b = _b[black_d]

quietly xtreg wage exper exper_bar school union_d married_d health_d black_d hisp_d, re
scalar mnd_w = _b[exper]; scalar mnd_b2C = _b[exper_bar]
scalar mnd_s = _b[school]; scalar mnd_b = _b[black_d]

quietly mixed wage exper_win exper_bar school union_d married_d health_d ///
             black_d hisp_d || id: exper_win, reml covariance(unstructured)
scalar rs_w = _b[exper_win]; scalar rs_b2B = _b[exper_bar]
scalar rs_s = _b[school]; scalar rs_b = _b[black_d]

di _n %12s "Model"    ///
   %12s "exper_win" ///
   %12s "exper_bar" ///
   %10s "school"    ///
   %10s "black_d"

di %12s "OLS"      %12.4f ols_w   %12s "n/a"    %10.4f ols_s %10.4f ols_b
di %12s "FE"       %12.4f fe_w    %12s "n/a"    %10s "n/a"  %10s "n/a"
di %12s "Std RE"   %12.4f re_w    %12s "n/a"    %10.4f re_s  %10.4f re_b
di %12s "REWB"     %12.4f rewb_w  %12.4f rewb_b2B %10.4f rewb_s %10.4f rewb_b
di %12s "Mundlak"  %12.4f mnd_w   %12.4f mnd_b2C  %10.4f mnd_s  %10.4f mnd_b
di %12s "REWB+RS"  %12.4f rs_w    %12.4f rs_b2B   %10.4f rs_s   %10.4f rs_b

/* ── 12. Interpretation ──────────────────────────────────────
────────────────────────────────────────────────────────── */
di _n "================================================================="
di    "INTERPRETATION"
di    "================================================================="
di _n sprintf("Within effect (REWB exper_win ≈ FE): %.4f", rewb_w)
di    sprintf("  Each additional year of own experience raises wages by ~%.1f%%", rewb_w*100)
di    sprintf("  FE estimate: %.4f  (small diff: GLS vs OLS estimation)", fe_w)
di _n sprintf("Between effect (REWB exper_bar): %.4f", rewb_b2B)
di    sprintf("  Workers with 1 more year of AVERAGE experience earn ~%.1f%% more", rewb_b2B*100)
di    sprintf("  Smaller than within — between differences partly reflect")
di    sprintf("  unobserved stable traits confounded with experience")
di _n sprintf("OLS blends both: %.4f  (between %.4f and %.4f)", ols_w, rewb_b2B, rewb_w)
di    sprintf("  Uninterpretable — a mix of two distinct processes")
di _n sprintf("Education (school, time-invariant):")
di    sprintf("  FE:   cannot estimate (absorbed by person dummies)")
di    sprintf("  REWB: %.4f  (%.1f%% per year of schooling)", rewb_s, rewb_s*100)
di _n sprintf("Ethnicity (black_d, time-invariant):")
di    sprintf("  FE:   cannot estimate")
di    sprintf("  REWB: %.4f  (Black workers earn ~%.1f%% less, cond. on controls)", ///
              rewb_b, abs(rewb_b)*100)

di _n "-----------------------------------------------------------------"
di    "Replication resources:"
di    "  Bell & Jones (2015) .do files:"
di    "    https://doi.org/10.7910/DVN/23415"
di    "  Jordan & Philips (2023) qdmean (install from GitHub):"
di    "    cap ado uninstall qdmean"
di    "    net install qdmean, from(https://github.com/andyphilips/qdmean/raw/main/)"
di    "-----------------------------------------------------------------"
