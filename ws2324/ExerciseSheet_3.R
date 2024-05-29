library(ggplot2)
library(gridExtra)
library(effects)


# Replace "your_file.txt" with the actual path to your file
file_path <- "C:\\Users\\esual\\OneDrive\\Documents\\StaDS\\lmu_repo\\data\\glms_week_three_patent.txt"

# Read the data with read.table
patents <- read.table(file_path, header = TRUE, sep = "\t")

# View the structure of your data
str(patents)

# View the first few rows of your data
head(patents) 

# Excluding outliers for making more reliable statements
# about the effects of covariants

ggplot(patents, aes(y=nclaims, x=ncit)) + geom_point()
patents_old <- read.table(file_path, header = TRUE, sep = "\t")
patents <- patents[which(patents$nclaims <= 60),]
patents <- patents[which(patents$ncit <= 15),]

g1 <- ggplot(patents_old, aes(y=nclaims, x=ncit)) + geom_point()
g2 <- ggplot(patents, aes(y=nclaims, x=ncit)) + geom_point()
grid.arrange(g1, g2, nrow = 1)

# Centering Covariates
patents$yearc <- patents$year - mean(patents$year)
patents$ncountryc <- patents$ncountry - mean(patents$ncountry)
patents$nclaimsc <- patents$nclaims - mean(patents$nclaims)

# Modelling
glm1 <- glm(formula = ncit ~ yearc + ncountryc + nclaimsc + biopharm + ustwin
              + patus + patgsgr + opp, data = patents, family = "poisson")

summary(glm1)

confint(glm1)

glm1_nc <- glm(formula = ncit ~ year + ncountry + nclaims + biopharm + ustwin
               + patus + patgsgr + opp, data = patents, family = "poisson")

summary(glm1_nc)

# Relationship with the intercept
year_mean = mean(patents$year)
ncountry_mean = mean(patents$ncountry)
nclaims_mean = mean(patents$nclaims)
glm1$coefficients[1] - (glm1$coefficients[2]*year_mean + glm1$coefficients[3]*ncountry_mean +
                          glm1$coefficients[4]*nclaims_mean)

# Adding interaction term
bins <- c("biopharm", "opp", "ustwin", "patus", "patgsgr")
for (x in bins) patents[, x] = as.factor(patents[, x])

glm_i <- glm(ncit ~ ncountryc + yearc + patgsgr + nclaimsc * biopharm + patus + ustwin +
               opp, family=poisson(), data=patents)

summary(glm_i)

plot(predictorEffects(glm_i), partial.residuals=TRUE, lines=list(multiline=TRUE))

# Fitting a quasi poisson model (for dispersion parameter)
glm_quapoi <- glm(ncit ~ ncountryc + nclaimsc + patgsgr + yearc * biopharm + patus + ustwin + opp, 
                  family = "quasipoisson", data = patents)


summary(glm_quapoi)

# Fitting negative binomial (for dispersion parameter)
glm_neg_bin <- glm.nb(ncit ~ ncountryc + nclaimsc + patgsgr + yearc * biopharm + patus + ustwin +
                        opp, data=patents)
summary(glm_neg_bin)

?glm
