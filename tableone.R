
# PURPOSE -----------------------------------------------------------------

#' Create tableone for mortality paper 
#' 



# load libraries ----------------------------------------------------------

library(tidyverse)
library(tableone)

# load data ---------------------------------------------------------------

data_ = 
  read_csv("../../data/mortality_X_y.csv") %>% 
  rename(Sample = `...1`) 

sample_cleaned_names <- str_replace(data_$Sample,"-[0-9]{1,2}-[0-9]{3}[NC]{0,1}","")
sample_map <- 
  tibble(Sample = data_$Sample,Sample_clean = sample_cleaned_names)

data_$Sample <- sample_cleaned_names
data_ <- data_ %>% distinct()

data_$Cohort <- ''
for(x in data_$Sample){
  row <- data_[data_$Sample==x,]
  if(row$Cohort_Paris==1){
    data_[data_$Sample==x,"Cohort"] <- 'Paris'
  }
  if(row$Cohort_Cedar==1){
    data_[data_$Sample==x,"Cohort"] <- 'Cedar'
  }
  if(row$Cohort_Columbia==1){
    data_[data_$Sample==x,"Cohort"] <- 'Columbia'
  }
}

data_$Cardiomyopathy <- ""
for(x in data_$Sample){
  row <- data_[data_$Sample==x,]
  if(row$Cardiomyopathy_Adriamycin==1)data_[data_$Sample==x,"Cardiomyopathy"] <- 'Adriamycin'
  if(row$Cardiomyopathy_Amyloid==1)data_[data_$Sample==x,"Cardiomyopathy"] <- 'Amyloid'
  if(row$Cardiomyopathy_Chagas==1)data_[data_$Sample==x,"Cardiomyopathy"] <- 'Chagas'
  if(row$Cardiomyopathy_Congenital==1)data_[data_$Sample==x,"Cardiomyopathy"] <- 'Congenital'
  if(row$`Cardiomyopathy_Hypertrophic cardiomyopathy`==1)data_[data_$Sample==x,"Cardiomyopathy"] <- 'Hypertrophic'
  if(row$Cardiomyopathy_Idiopathic==1)data_[data_$Sample==x,"Cardiomyopathy"] <- 'Idiopathic'
  if(row$Cardiomyopathy_Myocarditis==1)data_[data_$Sample==x,"Cardiomyopathy"] <- 'Myocarditis'
  if(row$`Cardiomyopathy_Valvular Heart Disease`==1)data_[data_$Sample==x,"Cardiomyopathy"] <- 'Valvular Heart Disease'
  if(row$Cardiomyopathy_Viral==1)data_[data_$Sample==x,"Cardiomyopathy"] <- 'Viral'
}

data_$Blood_Type <- ""
for(x in data_$Sample){
  row <- data_[data_$Sample==x,]
  if(row$Blood_Type_A==1)data_[data_$Sample==x,"Blood_Type"] <- 'A'
  if(row$Blood_Type_AB==1)data_[data_$Sample==x,"Blood_Type"] <- 'AB'
  if(row$Blood_Type_B==1)data_[data_$Sample==x,"Blood_Type"] <- 'B'
  if(row$Blood_Type_O==1)data_[data_$Sample==x,"Blood_Type"] <- 'O'
}

data_$Died <- ""
data_[data_$expired==1,"Died"] <- "Died"
data_[data_$expired==0,"Died"] <- "Survived"


# Days to death -----------------------------------------------------------

data_ %>% 
  ggplot(aes(days_to_death)) +
  geom_histogram(binwidth = log10(7),color="black",fill="grey") +
  scale_x_continuous(
    breaks=c(1,7,28,365,365*2,365*5),
    labels=c("Day after Transplant","1\nWeek","1\nMonth","1\nYear","2\nYears","5\nyears"),
    trans = "log10") +
  scale_y_continuous(labels=scales::label_number(accuracy=1)) +
  xlab("Time since transplant") +
  ylab('Number of patients') +
  theme_classic() +
  theme(
    text = element_text(face="bold")
  )

tmp <- data_ %>% data.table::data.table()
tmp$PGD <- ifelse(tmp$PGD==1,"PGD","non-PGD")
g <- tmp %>% 
  ggplot(aes(factor(1),days_to_death)) +
  geom_hline(yintercept=1,color="red",linetype="dashed") +
  geom_hline(yintercept=7,color="red",linetype="dashed") +
  geom_hline(yintercept=365/12,color="red",linetype="dashed") +
  geom_hline(yintercept=365,color="red",linetype="dashed") +
  geom_hline(yintercept=365*2,color="red",linetype="dashed") +
  geom_hline(yintercept=365*5,color="red",linetype="dashed") +
  ggbeeswarm::geom_quasirandom() +
  theme_classic() +
  xlab("") +
  ylab("") +
  scale_y_continuous(trans="log2",breaks=c(1,7,365/12,365,365*2,365*5)) +
  theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    text = element_text(face="bold",size=16)
  )
ggsave("../../docs/imgs/survival_days_to_death.png",g,width=4,height=5)

g <- tmp %>% 
  ggplot(aes(factor(PGD),days_to_death)) +
  geom_hline(yintercept=1,color="red",linetype="dashed") +
  geom_hline(yintercept=7,color="red",linetype="dashed") +
  geom_hline(yintercept=365/12,color="red",linetype="dashed") +
  geom_hline(yintercept=365,color="red",linetype="dashed") +
  geom_hline(yintercept=365*2,color="red",linetype="dashed") +
  geom_hline(yintercept=365*5,color="red",linetype="dashed") +
  ggbeeswarm::geom_quasirandom() +
  theme_classic() +
  xlab("") +
  ylab("") +
  scale_y_continuous(trans="log2",breaks=c(1,7,365/12,365,365*2,365*5)) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    text = element_text(face="bold",size=16)
  )
ggsave("../../docs/imgs/survival_days_to_death_by_PGD.png",g,width=4,height=5)

tmp <- data_ %>% data.table::data.table()
tmp$death_period <- ""
tmp[days_to_death<=365*5]$death_period =  list("5 years")
tmp[days_to_death<=365*2]$death_period =  list("2 years")
tmp[days_to_death<=365]$death_period =  list("1 year")
tmp[days_to_death<=30]$death_period =  list("1 month")
tmp[days_to_death<=7]$death_period =  list("1 week")
tmp[days_to_death==1]$death_period =  list("1 day")
tmp$PGD <- ifelse(tmp$PGD==1,"PGD","non-PGD")
tmp[,.N,.(death_period,PGD)]
# TableOne ----------------------------------------------------------------

dems <- c("Age","Blood_Type","BMI","Donor_Age","Sex_F","History_Of_Tobacco_Use_Y","Diabetes_Y",'Cohort')
card <- c("Cardiomyopathy","Died","days_to_death")
tfactors <- c("Ischemic_Time","Mechanical_Support_Y","PGD")
hemo <- c("PA_Diastolic","PA_Systolic","PA_Mean","CVP","PCWP")
labs <- c("Creatinine","INR","TBILI","Sodium")
meds <- c("Antiarrhythmic_Use_Y","Beta_Blocker_Y","Prior_Inotrope_Y")
comps <- c("CVP/PCWP","MELDXI")
allVars <- c(dems,card,tfactors,hemo,labs,meds,comps)
myVars <- allVars

catVars <- c("Sex_F",
             "Blood_Type",
             "History_Of_Tobacco_Use_Y",
             "Diabetes_Y","Cardiomyopathy",
             "PGD","Mechanical_Support_Y",
             "Antiarrhythmic_Use_Y","Beta_Blocker_Y",
             "Prior_Inotrope_Y",
             'Cohort',"Died")

data_$Ischemic <- as.integer(data_$Cardiomyopathy_Ischemic==1)
data_$NonIschemic <- as.integer(data_$Cardiomyopathy_Ischemic==0,"NonIschemic")
card <- c(card,"Ischemic","NonIschemic")
allVars <- c(allVars,"Ischemic","NonIschemic")
myVars <- c(myVars,"Ischemic","NonIschemic")
catVars <- c(catVars,"Ischemic","NonIschemic")


write.csv(data_ %>% 
          select(-Ischemic,-NonIschemic),"../../data/mortality_X_y_cleaned_for_tableone.csv")

tab <- CreateTableOne(vars=myVars,test=T,data=data_,factorVars = catVars,strata="Cohort")

tab2 <- print(tab, exact = "cohort", quote = FALSE, noSpaces = TRUE, printToggle = FALSE)
## Save to a CSV file
write.csv(tab2,"../../data/Mortality_tableone_by_cohort.csv")


tab <- CreateTableOne(vars=myVars,test=T,data=data_,factorVars = catVars,strata="PGD")

tab2 <- print(tab, exact = "cohort", quote = FALSE, noSpaces = TRUE, printToggle = FALSE)
## Save to a CSV file
write.csv(tab2,"../../data/Mortality_tableone_by_pgd.csv")

tab <- CreateTableOne(vars=myVars,test=T,data=data_,factorVars = catVars,strata="Died")

tab2 <- print(tab, exact = "cohort", quote = FALSE, noSpaces = TRUE, printToggle = FALSE)
## Save to a CSV file
write.csv(tab2,"../../data/Mortality_tableone_by_died.csv")


