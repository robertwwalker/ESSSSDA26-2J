## ============================================================
##  Worked Example: Fixed Effects, Random Effects, and REWB
##  Bell, Fairbrother & Jones (2019) — Quality & Quantity 53:1051
##
##  Dataset: Cornwell & Rupert (1988) PSID Males Wages Panel
##    595→545 male household heads, 8 years (1980–1987), N=4,360
##    Source: plm package — data("Males", package="plm")
##    Citation: Vella & Verbeek (1998), Journal of Applied
##              Econometrics 13(2): 163–183
##
##  Research question:
##    How does work experience relate to log wages?
##    within persons (as they gain experience year-by-year)
##    versus between persons (do experienced workers earn more)?
##
##  The within and between effects differ substantially — OLS and
##  standard RE conflate them.  REWB recovers both simultaneously.
##
##  Packages required:
##    install.packages(c("lme4", "plm", "knitr"))
##  Optional (convenience wrapper):
##    install.packages("panelr")   # then: wbm(wage ~ exper | school, ...)
##
##  Replication resources:
##    Bell & Jones (2015) .do files: https://doi.org/10.7910/DVN/23415
##    Jordan & Philips (2023) qdmean: https://github.com/andyphilips/qdmean
## ============================================================

library(lme4)
library(plm)
library(knitr)

# ── 1. Load data ──────────────────────────────────────────────
data("Males", package = "plm")
d <- Males
cat("Dataset: plm::Males  (Cornwell-Rupert PSID, via Vella & Verbeek 1998)\n")
cat(sprintf("Individuals: %d  |  Years: 1980–1987  |  Rows: %d\n\n",
            length(unique(d$nr)), nrow(d)))

# ── 2. Prepare variables ──────────────────────────────────────
d$id       <- d$nr                               # person identifier
d$union_d  <- as.integer(d$union   == "yes")     # time-varying
d$married_d<- as.integer(d$married == "yes")     # time-varying
d$health_d <- as.integer(d$health  == "yes")     # time-varying (poor health)
d$black_d  <- as.integer(d$ethn    == "black")   # time-invariant
d$hisp_d   <- as.integer(d$ethn    == "hisp")    # time-invariant
# school: years of schooling — strictly time-invariant (within-SD = 0)

# ── 3. Decompose experience into within and between parts ─────
#   exper_bar: person-specific mean experience (between component)
#   exper_win: deviation from own mean       (within component)
d$exper_bar <- ave(d$exper, d$id)          # group mean
d$exper_win <- d$exper - d$exper_bar       # within-person deviation

cat("-----------------------------------------------------------------\n")
cat("Variable roles:\n")
cat("  wage      : log wage (outcome)\n")
cat("  exper     : work experience in years (time-varying)\n")
cat("  exper_bar : person mean of experience  — BETWEEN component\n")
cat("  exper_win : exper minus person mean    — WITHIN component\n")
cat("  school    : years of schooling (time-INVARIANT)\n")
cat("  black_d / hisp_d: ethnicity dummies (time-INVARIANT)\n")
cat("  union_d / married_d / health_d: time-varying controls\n\n")

# ── 4. Declare panel ──────────────────────────────────────────
pdat <- pdata.frame(d, index = c("id", "year"))

# ── 5. MODEL 1: Pooled OLS ────────────────────────────────────
# x coefficient on exper is an uninterpretable blend of within
# (person grows in experience) and between (experienced workers
# differ from inexperienced ones in many unobserved ways).
# SEs are also underestimated because clustering is ignored.
m_ols <- lm(wage ~ exper + school + union_d + married_d +
              health_d + black_d + hisp_d,
            data = d)

# ── 6. MODEL 2: Fixed Effects (FE) ────────────────────────────
# Demeaning removes all between-person variation, giving the
# within estimate.  school, black_d, hisp_d are dropped as
# time-invariant; FE cannot say anything about them.
m_fe  <- plm(wage ~ exper + union_d + married_d + health_d,
             data = pdat, model = "within", effect = "individual")

# ── 7. MODEL 3: Standard Random Effects (RE) ─────────────────
# Assumes within effect = between effect.  Tests (Hausman) will
# show this assumption is rejected for experience.
m_re  <- lmer(wage ~ exper + school + union_d + married_d +
                health_d + black_d + hisp_d + (1 | id),
              data = d, REML = FALSE)

# ── 8. MODEL 4: Within-Between RE (REWB) ─────────────────────
# Key model: separates within and between effects of experience.
# exper_win → within effect:  does your wage grow as you gain
#             experience year-by-year?  (= FE estimate, approx.)
# exper_bar → between effect: do workers with more average
#             experience earn systematically more?
# school, black_d, hisp_d now estimable (unlike FE).
m_rewb <- lmer(wage ~ exper_win + exper_bar + school +
                 union_d + married_d + health_d +
                 black_d + hisp_d + (1 | id),
               data = d, REML = FALSE)

# ── 9. MODEL 5: Mundlak parameterisation (equivalent to REWB) ─
# Raw exper + group mean exper_bar:
#   exper     → within effect (beta_W)
#   exper_bar → contextual effect (beta_2C, net of individual exper)
# Identity: beta_W + beta_2C = beta_2B  (REWB between effect)
m_mndl <- lmer(wage ~ exper + exper_bar + school +
                 union_d + married_d + health_d +
                 black_d + hisp_d + (1 | id),
               data = d, REML = FALSE)

# ── 10. MODEL 6: REWB with Random Slopes ─────────────────────
# Allows the within-experience effect to vary across individuals.
# Bell et al. (2019): omitting random slopes when they exist
# produces anti-conservative standard errors.
m_rs  <- lmer(wage ~ exper_win + exper_bar + school +
                union_d + married_d + health_d +
                black_d + hisp_d + (1 + exper_win | id),
              data = d, REML = FALSE,
              control = lmerControl(optimizer = "bobyqa"))

# ── 11. Hausman-style LRT: is within == between? ─────────────
cat("-----------------------------------------------------------------\n")
cat("Hausman-style LRT:  H0: within effect = between effect\n")
cat("(i.e., contextual effect beta_2C = 0)\n")
cat("-----------------------------------------------------------------\n")
lrt <- anova(m_re, m_rewb)
cat(sprintf("Chi-sq = %.2f, df = 1, p = %.4f\n",
            lrt$Chisq[2], lrt$`Pr(>Chisq)`[2]))
cat(if (lrt$`Pr(>Chisq)`[2] < 0.05)
      "→ Significant: within ≠ between effect of experience.\n"
    else
      "→ Not significant: effects are similar; standard RE acceptable.\n")

# ── 12. Mundlak identity check ────────────────────────────────
cat("\n-----------------------------------------------------------------\n")
cat("Mundlak identity:  beta_W + beta_2C = beta_2B\n")
cat("-----------------------------------------------------------------\n")
bW  <- fixef(m_mndl)["exper"]
b2C <- fixef(m_mndl)["exper_bar"]
b2B <- fixef(m_rewb)["exper_bar"]
cat(sprintf("  beta_W  (Mundlak, raw exper):      %7.5f\n", bW))
cat(sprintf("  beta_2C (Mundlak, exper_bar):      %7.5f\n", b2C))
cat(sprintf("  Sum:                               %7.5f\n", bW + b2C))
cat(sprintf("  beta_2B (REWB,    exper_bar):      %7.5f  ✓\n\n", b2B))

# ── 13. Random slopes variance components ────────────────────
cat("-----------------------------------------------------------------\n")
cat("Random effects variance (REWB + Random Slopes)\n")
cat("-----------------------------------------------------------------\n")
vc <- as.data.frame(VarCorr(m_rs))[, c("grp","var1","var2","vcov","sdcor")]
print(vc); cat("\n")

# ── 14. Summary comparison table ─────────────────────────────
cat("=================================================================\n")
cat("SUMMARY: Coefficient on experience across models\n")
cat("=================================================================\n")

exper_within <- c(
  coef(m_ols)["exper"],
  as.numeric(coef(m_fe)["exper"]),
  fixef(m_re)["exper"],
  fixef(m_rewb)["exper_win"],
  fixef(m_mndl)["exper"],
  fixef(m_rs)["exper_win"]
)
exper_between <- c(NA, NA, NA,
  fixef(m_rewb)["exper_bar"],
  fixef(m_mndl)["exper_bar"],
  fixef(m_rs)["exper_bar"])
school_coef <- c(
  coef(m_ols)["school"],
  NA,
  fixef(m_re)["school"],
  fixef(m_rewb)["school"],
  fixef(m_mndl)["school"],
  fixef(m_rs)["school"])
black_coef <- c(
  coef(m_ols)["black_d"],
  NA,
  fixef(m_re)["black_d"],
  fixef(m_rewb)["black_d"],
  fixef(m_mndl)["black_d"],
  fixef(m_rs)["black_d"])

res <- data.frame(
  Model        = c("OLS","FE","Std RE","REWB","Mundlak","REWB+RS"),
  `exper (within/blend)` = round(exper_within, 4),
  `exper_bar (between/contextual)` = round(exper_between, 4),
  `school`  = round(school_coef, 4),
  `black_d` = round(black_coef, 4),
  check.names = FALSE)

print(kable(res, format = "simple"))

cat("\n=================================================================\n")
cat("INTERPRETATION\n")
cat("=================================================================\n")
fe_b    <- as.numeric(coef(m_fe)["exper"])
rewb_w  <- fixef(m_rewb)["exper_win"]
rewb_b  <- fixef(m_rewb)["exper_bar"]
cat(sprintf(
  "Within effect  (REWB exper_win ≈ FE): %.4f
  → Each additional year of own experience raises wages by ~%.1f%%
  → FE estimate: %.4f  (small diff due to GLS vs OLS estimation)\n\n",
  rewb_w, rewb_w*100, fe_b))
cat(sprintf(
  "Between effect (REWB exper_bar):       %.4f
  → Workers with 1 more year of AVERAGE experience earn ~%.1f%% more
  → Smaller than within effect — between differences partly reflect
    unobserved stable traits (ability, motivation) confounded with
    experience accumulation\n\n",
  rewb_b, rewb_b*100))
cat(sprintf(
  "OLS blends both: %.4f  (between %.4f and %.4f)
  → Uninterpretable — a mix of two distinct processes\n\n",
  coef(m_ols)["exper"], rewb_b, rewb_w))
cat(sprintf(
  "Education (school, time-invariant):
  → FE:  cannot estimate (absorbed by person dummies)
  → REWB: %.4f  (≈%.1f%% per year of schooling)\n\n",
  fixef(m_rewb)["school"], fixef(m_rewb)["school"]*100))
cat(sprintf(
  "Ethnicity (black_d, time-invariant):
  → FE:  cannot estimate (absorbed by person dummies)
  → REWB: %.4f  (Black workers earn ~%.1f%% less, conditional on\n
           experience, education, union etc.)\n",
  fixef(m_rewb)["black_d"], abs(fixef(m_rewb)["black_d"])*100))
