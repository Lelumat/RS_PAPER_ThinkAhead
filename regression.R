library(lme4)

#SUCCESS RATE

dfLogit <- read.csv("dfLogit.csv")

#Full model
logit_fullModel <- glmer(SUCCESSFUL ~ DepthID + Level + DepthID : Level + (1|SubjID), data = dfLogit, family = binomial)

#Reduced models
logit_modelNoRandom <- glm(SUCCESSFUL ~ DepthID + Level + DepthID : Level, data = dfLogit, family = binomial)
glmer_noIntercept <- glmer(SUCCESSFUL ~ 0 + DepthID + Level + DepthID : Level + (1|SubjID), data = dfLogit, family = binomial)
glmer_modelNoInteraction <- glmer(SUCCESSFUL ~ DepthID + Level + (1|SubjID), data = dfLogit, family = binomial)
glmer_DepthInt <- glmer(SUCCESSFUL ~ Level + Level:DepthID + (1|SubjID), data = dfLogit, family = binomial)
glmer_LevelInt <- glmer(SUCCESSFUL ~ DepthID + Level:DepthID + (1|SubjID), data = dfLogit, family = binomial)

cat("\n\n Is the random effect significant?: \n\n")
anova(logit_fullModel, logit_modelNoRandom, test = "LRT")
cat("\n\n Is the intercept significant?: \n\n")
anova(logit_fullModel, glmer_noIntercept, test = "LRT")
cat("\n\n Is the interaction significant?: \n\n")
anova(logit_fullModel, glmer_modelNoInteraction, test = "LRT")
cat("\n\n Is there a main effect of depth?: \n\n")
anova(logit_fullModel, glmer_DepthInt, test = "LRT")
cat("\n\n Is there a main effect of level?: \n\n")
anova(logit_fullModel, glmer_LevelInt, test = "LRT")
cat("\n\n Summary of the full model: \n\n")

summary(logit_fullModel)



#TOTAL TIME
dfTime <- read.csv("dfTime.csv")

#FUll model
time_fullModel <- lmer(TOTAL_TIME ~ DepthID * Level + (1|SubjID), data = dfTime)

#Reduced models
time_modelNoRandom <- lm(TOTAL_TIME ~ DepthID + Level + DepthID : Level, data = dfTime)
time_modelNoIntercept <- lmer(TOTAL_TIME ~ 0 + DepthID * Level + (1|SubjID), data = dfTime)
time_modelNoInteraction <- lmer(TOTAL_TIME ~ DepthID + Level + (1|SubjID), data = dfTime)
time_DepthInt <- lmer(TOTAL_TIME ~ Level + Level:DepthID + (1|SubjID), data = dfTime)
time_LevelInt <- lmer(TOTAL_TIME ~ DepthID + Level:DepthID + (1|SubjID), data = dfTime)


cat("\n\n Is the random effect significant?: \n\n")
anova(time_fullModel, time_modelNoRandom, test = "LRT")
cat("\n\n Is the intercept significant?: \n\n")
anova(time_fullModel, time_modelNoIntercept, test = "LRT")
cat("\n\n Is the interaction significant?: \n\n")
anova(time_fullModel, time_modelNoInteraction, test = "LRT")
cat("\n\n Is there a main effect of depth?: \n\n")
anova(time_fullModel, time_DepthInt, test = "LRT")
cat("\n\n Is there a main effect of level?: \n\n")
anova(time_fullModel, time_LevelInt, test = "LRT")
cat("\n\n Summary of the full model: \n\n")
summary(time_fullModel)


dfBacktrack <- read.csv("dfBacktrack.csv")
#Make a box plot of the data where on the x-axis we have the DepthID and on the y-axis the N_BACKTRACK


#FUll model
backtrack_fullModel <- lmer(N_BACKTRACK ~ DepthID * Level + (1|SubjID), data = dfBacktrack)

#Reduced models
backtrack_modelNoRandom <- lm(N_BACKTRACK ~ DepthID + Level + DepthID : Level, data = dfBacktrack)
backtrack_modelNoIntercept <- lmer(N_BACKTRACK ~ 0 + DepthID * Level + (1|SubjID), data = dfBacktrack)
backtrack_modelNoInteraction <- lmer(N_BACKTRACK ~ DepthID + Level + (1|SubjID), data = dfBacktrack)
backtrack_DepthInt <- lmer(N_BACKTRACK ~ Level + Level:DepthID + (1|SubjID), data = dfBacktrack)
backtrack_LevelInt <- lmer(N_BACKTRACK ~ DepthID + Level:DepthID + (1|SubjID), data = dfBacktrack)


cat("\n\n Is the random effect significant?: \n\n")
anova(backtrack_fullModel, backtrack_modelNoRandom, test = "LRT")
cat("\n\n Is the intercept significant?: \n\n")
anova(backtrack_fullModel, backtrack_modelNoIntercept, test = "LRT")
cat("\n\n Is the interaction significant?: \n\n")
anova(backtrack_fullModel, backtrack_modelNoInteraction, test = "LRT")
cat("\n\n Is there a main effect of depth?: \n\n")
anova(backtrack_fullModel, backtrack_DepthInt, test = "LRT")
cat("\n\n Is there a main effect of level?: \n\n")
anova(backtrack_fullModel, backtrack_LevelInt, test = "LRT")
cat("\n\n Summary of the full model: \n\n")
summary(backtrack_fullModel)