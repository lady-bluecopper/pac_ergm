# Genera il dataset ukFaculty in CSV + la ground-truth per validare il Python.
# Run-once, da tenere nel repo per riproducibilità.
need <- c("lolog", "ergm")
ins  <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
if (length(ins)) install.packages(ins, repos = "https://cloud.r-project.org")

suppressPackageStartupMessages({ library(lolog); library(ergm); library(network) })
data(ukFaculty)                                            # il dataset vive in lolog
delete.vertices(ukFaculty, which(is.na(ukFaculty %v% "Group")))   # come la vignetta
stopifnot(is.directed(ukFaculty))
n <- network.size(ukFaculty)
cat("nodi:", n, " archi diretti:", network.edgecount(ukFaculty), "\n")

# --- attributi nodali (id 0-based) -> PERMANENTE ---
nodes <- data.frame(id = 0:(n-1),
                    Group  = ukFaculty %v% "Group",
                    GroupC = ukFaculty %v% "GroupC")
write.csv(nodes, "data/ukfaculty_nodes.csv", row.names = FALSE)
cat("\n== GroupC ==\n"); print(table(nodes$GroupC))
cat("livelli ordinati (ordine ergm):", paste(sort(unique(as.character(nodes$GroupC)))), "\n")

# --- edge list diretta (0-based) -> PERMANENTE ---
el <- as.edgelist(ukFaculty)                              # 1-based (tail, head)
write.csv(data.frame(from = el[,1]-1, to = el[,2]-1),
          "data/ukfaculty_edges.csv", row.names = FALSE)

# --- GROUND TRUTH: g_obs autorevole (il giudice del Passo 6) ---
f <- ukFaculty ~ edges + nodematch("GroupC") +
       nodefactor("GroupC", base = 3) + gwesp(decay = .25, fixed = TRUE)
gobs <- summary(f)
cat("\n== summary(~ ...)  = g_obs autorevole ==\n"); print(gobs)
write.csv(data.frame(name = names(gobs), value = as.numeric(gobs)),
          "data/ukfaculty_gobs_R.csv", row.names = FALSE)

# --- validazione profonda: change statistics per ogni coppia ---
mpl <- ergmMPLE(f, output = "matrix")
write.csv(cbind(response = mpl$response, weights = mpl$weights, mpl$predictor),
          "data/ukfaculty_mple_R.csv", row.names = FALSE)
cat("\n== nomi termini (quali nodefactor.GroupC.* restano) ==\n")
print(colnames(mpl$predictor))