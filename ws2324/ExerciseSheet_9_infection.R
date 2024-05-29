load("./StaDS/lmu_repo/data/Infection.RData")
Infect$date<-as.Date(Infect$date)
Infect$Type<-as.factor(Infect$Type)
summary(Infect)
str(Infect)

# Start off with a generalized additive model in which you aim to estimate the number of COVID-19
# patients in the ICUs in dependence of the infections of the logged average infection rates of people
# aged 35 and above. Explore, through the penalized splines, whether a generalized additive model
# is necessary. This does not require a formal hypothesis test. Use the gam() function from the
# mgcv-package

any(is.na(Infect))

library(mgcv)

model_1<-mgcv::gam(faelle_covid_aktuell_avg ~s(log_G_4_lag_w_1, bs="ps")+
                     s(log_G_5_lag_w_1, bs="ps")+
                     s(log_G_6_lag_w_1, bs="ps"),data=Infect, family=poisson())
summary.gam(model_1)

# wait, shouldn’t we take quasi-poisson instead of poisson?
model_qp<-mgcv::gam(faelle_covid_aktuell_avg ~s(log_G_4_lag_w_1, bs="ps")+
                      s(log_G_5_lag_w_1, bs="ps")+
                      s(log_G_6_lag_w_1, bs="ps"),data=Infect, family=quasipoisson())
summary.gam(model_qp)
