extub_event <- read.csv("./StaDS/lmu_repo/data/extub_event.csv")
head(extub_event)
table(extub_event$outcome)
class(extub_event$outcome)
hist(extub_event$time)

library("ggsurvfit")
library(cowplot)
library(survdiff)
library(survival)

g1<-survfit2(Surv(time, outcome) ~ 1, data = extub_event) %>%
  ggsurvfit() +
  labs(
    x = "Days",
    y = "Overall survival probability"
  ) +
  geom_hline(aes(yintercept=0.2), col="red")+
  add_confidence_interval()
g2<-survfit2(Surv(time, outcome) ~ as.factor(sex), data = extub_event) %>%
  ggsurvfit() +
  labs(
    x = "Days",
    y = "Overall survival probability"
  ) +
  geom_hline(aes(yintercept=0.2), col="red")+
  add_confidence_interval() +
  theme(legend.position = c(0.8, 0.8))
plot_grid(g1, g2, nrow=1)

# In the right panel of the second plot, we see that there is some difference between males and
#females. sex = 1 seems to have a somewhat higher survival probability across all time points.
# We could test whether this significance is significant, using the log-rank test

log_rank_test <- survdiff(Surv(time, outcome) ~ as.factor(sex), data = extub_event)
log_rank_test

# Write out the model equation for modeling the hazard rate as a function of the patient’s sex,
# admission type, and SAPS score using a Cox Proportional Hazards model. Fit the model in
# R, using the function coxph from the package survival

cox <- survival::coxph(
  Surv(time, outcome) ~ sex + type + SAPSadmission,
  data = extub_event,
  x = TRUE
)

summary(cox)

# If instead we wish to model the data using an Accelerated Failure Time (AFT) model with
# Weibull-distributed T, which distributional assumption are we imposing on the error terms and
# on the event times? 

weibull <- survreg(
  Surv(time, outcome) ~ sex + type + SAPSadmission,
  dist = "weibull", data = extub_event
)

summary(weibull)

# according to the parametrization used by the survreg function, we have
# alpha = 1/weibull$scale.
# The hazard ratio corresponding to β2 is then exp(−αβ2) = 0.6248.

exp(-1/weibull$scale * coef(weibull)[3])

# For instance, we might have different baseline hazards according to admission type:
cox_strata <- survival::coxph(Surv(time, outcome) ~ strata(type) + sex + SAPSadmission,
                              data = extub_event, x = TRUE)
summary(cox_strata)

cox_strata_df <- broom::tidy(survfit(cox_strata))
ggplot(cox_strata_df, aes(x = time, y = estimate, col = strata)) +
  geom_step()

# The effect of the SAPS score at admission on the extubation risk might not necessarily be a linear
# function. Using the function pspline from the package survival, include the SAPS admission
# score as smooth, penalized function.
cox_add <- survival::coxph(
  Surv(time, outcome) ~ sex + type + pspline(SAPSadmission, df=3),
  data = extub_event, x = TRUE)
summary(cox_add)

library(pammtools)
cox_add_df <- make_newdata(
  extub_event, SAPSadmission = seq_range(SAPSadmission, 100))
terms <- predict(cox_add, newdata = cox_add_df, type = "terms")
cox_add_df$spline <- terms[, 3]
ggplot(cox_add_df, aes(x = SAPSadmission, y = spline)) +
  geom_line()


termplot(cox_add, term = 3, se = TRUE, col.term = 1, col.se = 1) # alternative

extub_event$SAPSadmission[1:10]
