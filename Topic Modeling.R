#
#
# DESCRIPTION:
#
#       This project involves "Topic Modeling Analysis of Terrorist Organizations as portrayed in traditional media"
# 
# DATASET:
#
#       The dataset which was downloaded from Fortiva - a leading global provider 
#
#       of economic and financial information - were collected in 2017 and contains Articles from
#     
#       the Wall Street Journal and New York Times.
#
#


# Import Libraries

library(NLP)
library(tm)
library(stm)
library(topicmodels)
library(tidytext)
library(dplyr)
library(ggplot2)
library(stringr)
library(cluster)
library(topicmodels)
library(tokenizers)
library(quanteda)


#
# Aproach 1 Latent Dirichlet Allocation or LDA
#

#  Create List from the folder

Articles <- Corpus(DirSource("Articles"))


#  phase 1
#
# separating articles in each txt file
#

splitted_article=list()
for (i in (1:length(Articles))){
  splitted_article[[i]]=(strsplit(as.String(Articles[[i]][[1]]),"All Rights Reserved"))
}

#  phase2 
#
# the first one should be removed 
#
# because it has only basic information (meta data) about first article
#
# Also tokenizing each single article


splitted_article2<- NULL
splitted_article2[cbind(1:18)]<-list(rep(list(),2))
for (i in (1:length(splitted_article))){
  for (j in 2:length(splitted_article[[i]][[1]])){
  splitted_article2[[i]][[j-1]]= (tokenize_words(as.character(splitted_article[[i]][[1]][j])))[[1]]
  }}


#  phase3
#
#  Now we remove meta data based on two key word 
#
#  last occurance of "by" or last occurance of "document"
#
#  and eliminate everything after that

splitted_article3<- NULL
splitted_article3[cbind(1:18)]<-list(rep(list(),2))
for (i in (1:18)){
  for (j in 1:(length(splitted_article2[[i]]))){
    a<-max(which(splitted_article2[[i]][[j]]=="by"))
    b<-ifelse(is.infinite(a),max(which(splitted_article2[[i]][[j]]=="document")),a)
    splitted_article3[[i]][[j]]<-list()
    splitted_article3[[i]][[j]][[1]]<- splitted_article2[[i]][[j]][1:b-1]
  }}


# phase4
#
# Create a new folder namely "New_Article"

dir.create(file.path(getwd(), "New_Article"), showWarnings = FALSE)

# Set it as working directory

setwd(file.path(getwd(), "New_Article"))

# Then I save all articles as a string to a separate file

for (i in 1:length(splitted_article3)){
  for (j in 1:length(splitted_article3[[i]])){
    sink(paste0("Article",i,j,".txt"))
    #Using "iconv" function to delete/translate non-english terms
    print(as.String(iconv(splitted_article3[[i]][[j]][[1]],
                          from = "LATIN1", to="ASCII")))
    sink()
  }
}

#   Back to original directory

setwd("/project3")

# And read each single article again!

New_Article <- Corpus(DirSource("New_Article",encoding = "UTF-8"))

#Transform to lower case

Clean_Articles <- tm_map(New_Article,content_transformer(tolower))

#remove punctuation

Clean_Articles <- tm_map(Clean_Articles, removePunctuation)

#Strip digits

Clean_Articles <- tm_map(Clean_Articles, removeNumbers)

#remove whitespace

Clean_Articles <- tm_map(Clean_Articles, stripWhitespace)

#remove stopwords

Clean_Articles <- tm_map(Clean_Articles, removeWords, stopwords("english"))

#New stop words

myStopwords2 <- c("can", "say","one","way","use",
                 "also","howev","tell","will",
                 "much","need","take","tend","even",
                 "like","particular","rather","said",
                 "get","well","make","ask","come","end",
                 "first","two","help","often","may",
                 "might","see","someth","thing","point",
                 "post","look","right","now","think","'ve ",
                 "'re ","new","york","wall","street",
                 "journal","online","newyork","wsjo",
                 "english","copyright","inc","dow","jone",
                 "company","copy","right","nytimescom","nytfeed",
                 "january","february","march","april","may","june",
                 "july","august","september","october","november",
                 "december","nytf","nytfeededh","nytfeededhjh",
                 "time","compani","copyright","english","told","dont","nytfeededl",
                 "edit","many","day","month","year",
                 "monday","tuesday","wednesday","thursday",
                 "friday","saturday","sunday","white","house",
                 "whitehouse",letters,"week","work",
                 "want","just","man","woman","know","ac",
                 "mr","list",
                 "na",##It must be there because of iconv
                 "nytf.com","wsj.com","aw")

#remove mystopwords

Clean_Articles <- tm_map(Clean_Articles, removeWords, myStopwords2)


#Stem document

Clean_Articles <- tm_map(Clean_Articles,stemDocument)

##Again cleaning after stemming! To make sure nothing left!
#remove punctuation
 
Clean_Articles <- tm_map(Clean_Articles, removePunctuation)

#Strip digits

Clean_Articles <- tm_map(Clean_Articles, removeNumbers)

#remove whitespace

Clean_Articles <- tm_map(Clean_Articles, stripWhitespace)

#remove stopwords

Clean_Articles <- tm_map(Clean_Articles, removeWords, stopwords("english"))

#remove mystopwords

Clean_Articles <- tm_map(Clean_Articles, removeWords, myStopwords2)

#To make sure isis and islamic state is exist in the file!!

writeLines(as.character(Clean_Articles[[190]]))
writeLines(as.character(Clean_Articles[[3]]))

####################### End of preprocessing

# Creating Document Term Matrix

dtm <- DocumentTermMatrix(Clean_Articles)

#Exploration of dtm

sink("stm.txt")
inspect(dtm)
sink()

#Turn it to matrix

m<-as.matrix(dtm)
Ins<-as.matrix(inspect(dtm))
write.csv(Ins,"Ins.csv")

#shorten rownames for display purposes

rownames(m) <- paste(substring(rownames(m),1,3),rep("..",nrow(m)),
                     substring(rownames(m),
                               nchar(rownames(m))-12,nchar(rownames(m))-4))

#Visualization

textplot_wordcloud(dfm(corpus(Clean_Articles)))

#Choosing k topic for the articles

start_time <- Sys.time()
topic_model<-LDA(m, k=6,method = "VEM", control = list(seed = 100))
end_time <- Sys.time()
print(paste("Processing time is",end_time-start_time ))

topics <- tidy(topic_model, matrix = "beta")

#Finding top 10 words of each group

top_terms <- 
  topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

#Visualization of top words

top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  ylab("Value of Beta")+
  xlab("Top 10 words")+
  coord_flip()




