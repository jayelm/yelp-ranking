library(tidyverse)
library(igraph)
library(data.table)

ms_graph = read.dot('./dataset_processed/dedup_tinier.dot') %>%
  graph.adjacency

png('img/graph.png', 2000, 2000)
plot(ms_graph)
dev.off()
