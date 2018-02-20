library(tidyverse)
library(igraph)
library(data.table)

ms_graph = fread('dataset_processed/matches.csv') %>%
  tbl_df %>%
  filter(win != 0) %>%
  select(b1, b2) %>%
  as.matrix
ms_graph = ms_graph + 1
ms_graph = graph_from_edgelist(ms_graph, directed = FALSE)

png('img/graph.png', 2000, 2000)
plot(ms_graph)
dev.off()
