library(lme4)
library(lmerTest)

#SUCCESS RATE

dfLogit <- read.csv("dfLogit.csv")

#Full model
logit_fullModel <- glmer(SUCCESSFUL ~ DepthID + Level + DepthID : Level + (1|SubjID), data = dfLogit, family = binomial)
#Summary of the full model
summary(logit_fullModel)


#TOTAL TIME
dfTime <- read.csv("dfTime.csv")
#FUll model
time_fullModel <- lmer(TOTAL_TIME ~ DepthID * Level + (1|SubjID), data = dfTime)
summary(time_fullModel)


dfBacktrack <- read.csv("dfBacktrack.csv")
#FUll model
backtrack_fullModel <- lmer(N_BACKTRACK ~ DepthID * Level + (1|SubjID), data = dfBacktrack)
summary(backtrack_fullModel)