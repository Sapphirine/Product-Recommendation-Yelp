library(shiny)
library(leaflet)
library(dplyr)
library(tidyr)
library(tidyverse)


df <- read.csv("top_10_vertival_4944.csv",header = TRUE,stringsAsFactors = FALSE)


df <- tidyr::separate(data=df,
                      into=c("latitude", "longitude"),
                      remove=FALSE)

df$latitude <- stringr::str_replace_all(df$latitude, "[(]", "")
df$longitude <- stringr::str_replace_all(df$longitude, "[)]", "")


df$latitude <- as.numeric(df$latitude)
df$longitude <- as.numeric(df$longitude)
saveRDS(df, "./data.rds")


sample_data <- df[c(1:1000),]
saveRDS(sample_data, "./sample_data.rds")