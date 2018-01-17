## Loading all Required packages
install.packages("qdap")
library(qdap)
library(tm)
install.packages("RWeka")
library(RWeka)
install.packages("wordcloud")
library(wordcloud)
library(plotrix)
library(readr)

##======Importing datasets ----- Reading csv files of Amazon and Google=================

amzn_reviews <- read.csv("https://assets.datacamp.com/production/course_935/datasets/500_amzn.csv")
View(amzn_reviews)
goog_reviews <- read.csv("https://assets.datacamp.com/production/course_935/datasets/500_goog.csv")
View(goog_reviews)

##=============================Examining Text Sources===================================
## Preview of structure of amzn data
str(amzn_reviews)

## Defining two vectors for postive and negative reviews separately
amzn_pros <- amzn_reviews$pros
amzn_cons <- amzn_reviews$cons

## Preview of structure of goog data
str(goog_reviews)

## Defining two vectors for postive and negative reviews of goog separately
goog_pros <- goog_reviews$pros
goog_cons <- goog_reviews$cons

##=============================Text Organization======================================

## Defining function qdap_clean() for cleaning
qdap_clean <- function(x){
  x <- replace_abbreviation(x)
  x <- replace_contraction(x)
  x <- replace_number(x)
  x <- replace_ordinal(x)
  x <- replace_symbol(x)
  x <- tolower(x)
  return(x)
}

## Defining function tm_clean for cleaning using tm package
tm_clean <- function(corpus){
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removeWords, c(stopwords("en"),"Google","Amazon","company"))
  return(corpus)
}

## Cleaning amazon reviews
amzn_pros <- qdap_clean(amzn_pros)
amzn_pros <- na.omit(amzn_pros)
amzn_cons <- qdap_clean(amzn_cons)
amzn_cons <- na.omit(amzn_cons)

az_p_corp <- VCorpus(VectorSource(amzn_pros))
az_c_corp <- VCorpus(VectorSource(amzn_cons))

amzn_pros_corp <- tm_clean(az_p_corp)
amzn_cons_corp <- tm_clean(az_c_corp)

## Cleaning google reviews
goog_pros <- qdap_clean(goog_pros)
goog_pros <- na.omit(goog_pros)
goog_cons <- qdap_clean(goog_cons)
goog_cons <- na.omit(goog_cons)

goog_p_corp <- VCorpus(VectorSource(goog_pros))
goog_c_corp <- VCorpus(VectorSource(goog_cons))

goog_pros_corp <- tm_clean(goog_p_corp)
goog_cons_corp <- tm_clean(goog_c_corp)

##==================Feature Extraction and Analysis=========================

tokenizer <- function(x)
  NGramTokenizer(x, Weka_control(min = 2, max = 2))

## Creating bigram
amzn_p_tdm <- TermDocumentMatrix(amzn_pros_corp, control = list(tokenize = tokenizer))
amzn_p_tdm_m <- as.matrix(amzn_p_tdm)
## Obtaining term frequencies
amzn_p_freq <- rowSums(amzn_p_tdm_m)
## Creating word cloud for amazon positive reviews
wordcloud(names(amzn_p_freq), amzn_p_freq, max.words = 25, color = "blue")

## Creating wordcloud for amazon negative reviews
amzn_c_tdm <- TermDocumentMatrix(amzn_cons_corp, control = list(tokenizer = tokenizer))
amzn_c_tdm_m <- as.matrix(amzn_c_tdm)
amzn_c_freq <- rowSums(amzn_c_tdm_m)
wordcloud(names(amzn_c_freq), amzn_c_freq, max.words = 25, color = "red")

## Dendrogram to check connection between phrases for amzn_cons
amzn_c_tdm
amzn_c_tdm2 <- removeSparseTerms(amzn_c_tdm, sparse = 0.993)
hc <- hclust(dist(amzn_c_tdm2, method = "euclidean"), method = "complete")
plot(hc)

## Checkin word association for positive comments of amazon positive comments
## Sorting term frequencies of positive comments
term_frequency <- sort(amzn_p_freq, decreasing = TRUE)
term_frequency[1:5]
findAssocs(amzn_p_tdm, "fast paced", 0.2)

## Making comparison cloud of google reviews for comparison to Amazon reviews
goog_pros_collapse <- paste(goog_pros, collapse = " ")
goog_cons_collapse <- paste(goog_cons, collapse = " ")
all_goog_reviews <- c(goog_pros_collapse, goog_cons_collapse)

## Creating Corpus and cleaning all Google reviews
all_goog_corpus <- VCorpus(VectorSource(all_goog_reviews))
all_goog_corp <- tm_clean(all_goog_corpus)
all_goog_tdm <- TermDocumentMatrix(all_goog_corp)
colnames(all_goog_tdm) <- c("Goog_Pros", "Goog_Cons")
all_goog_m <- as.matrix(all_goog_tdm)
comparison.cloud(all_goog_m, colors = c("#F44336", "#2196f3"), max.words = 100)

## Plotting Pyramid plot for positive reviews of Amazon and Google to see differences
## between any shared bigrams
amzn_pros_collapse <- paste(amzn_pros, collapse = " ")
all_pros_reviews <- c(amzn_pros_collapse, goog_pros_collapse)

## Creating Corpus and cleaning all positive reviews of Amazon and Google
all_pros_corpus <- VCorpus(VectorSource(all_pros_reviews))
all_pros_corp <- tm_clean(all_pros_corpus)
all_pros_tdm <- TermDocumentMatrix(all_pros_corp, control = list(tokenize = tokenizer))

colnames(all_pros_tdm) <- c("Amazon_Pros", "Google_Pros")
all_pros_m <- as.matrix(all_pros_tdm)
head(all_pros_m)

#=========== Identifying bigrams that occur in both comapnies's pro reviews ==============
common_words <- subset(all_pros_m, all_pros_m[, 1]>0 & all_pros_m[, 2]>0)
difference <- abs(common_words[,1] - common_words[,2])
common_words <- cbind(common_words, difference)
common_words <- common_words[order(common_words[,3], decreasing = TRUE), ]

## Creating a data frame of the top 15 bigrams
top15_df <- data.frame(x = common_words[1:15, 1], y = common_words[1:15, 2], labels = rownames(common_words[1:15, ]) )
pyramid.plot(top15_df$x, top15_df$y, labels = top15_df$labels, 
             gap = 12, top.labels = c("Amzn", "Pro Words", "Google"), 
             main = "Words in Common", unit = NULL)

## Plotting Pyramid plot for negative reviews of Amazon and Google to see differences
## between any shared bigrams
amzn_cons_collapse <- paste(amzn_cons, collapse = " ")
all_cons_reviews <- c(amzn_cons_collapse, goog_cons_collapse)

## Creating Corpus and cleaning all negative reviews of Amazon and Google
all_cons_corpus <- VCorpus(VectorSource(all_cons_reviews))
all_cons_corp <- tm_clean(all_cons_corpus)
all_cons_tdm <- TermDocumentMatrix(all_cons_corp, control = list(tokenize = tokenizer))

colnames(all_cons_tdm) <- c("Amazon_Cons", "Google_Cons")
all_cons_m <- as.matrix(all_cons_tdm)
head(all_cons_m)

#=========== Identifying bigrams that occur in both comapnies's pro reviews ==============
common_words <- subset(all_cons_m, all_cons_m[, 1]>0 & all_cons_m[, 2]>0)
difference <- abs(common_words[,1] - common_words[,2])
common_words <- cbind(common_words, difference)
common_words <- common_words[order(common_words[,3], decreasing = TRUE), ]

## Creating a data frame of the top 15 bigrams
top15_df <- data.frame(x = common_words[1:15, 1], 
                      y = common_words[1:15, 2], 
                      labels = rownames(common_words[1:15, ]) )

pyramid.plot(top15_df$x, top15_df$y, labels = top15_df$labels, 
             gap = 12, top.labels = c("Amzn", "Cons Words", "Google"), 
             main = "Words in Common", unit = NULL)

