
suppressMessages({library(network); library(lolog)})
a <- commandArgs(trailingOnly = TRUE)
edges_csv <- a[1]; cov_dir <- a[2]; nsim <- as.integer(a[3]); out_prefix <- a[4]

ed <- read.csv(edges_csv)
rdm <- function(f) as.matrix(read.csv(file.path(cov_dir, f), header = FALSE))
m_use <- rdm("use.csv"); m_team <- rdm("team.csv")
m_flr <- rdm("floor.csv"); m_met <- rdm("metric.csv")
n <- nrow(m_use)

net <- network.initialize(n, directed = TRUE)
net[as.matrix(ed[, c("from", "to")]) + 1] <- 1

form <- net ~ edges + mutual + gwesp(0.5) +
        star(c(2, 3), direction = "out") + star(c(2, 3), direction = "in") +
        edgeCov(m_use, "use") + edgeCov(m_team, "team") +
        edgeCov(m_flr, "floor") + edgeCov(m_met, "metric")

status <- "CONVERGED"
t0 <- Sys.time()
set.seed(1)
fit <- tryCatch(lolog(form, verbose = 0),
                error = function(e) { status <<- paste("ERROR:", conditionMessage(e)); NULL })
secs <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
if (is.null(fit)) {
  writeLines(c(status, paste("fit seconds:", secs)), paste0(out_prefix, "_status.txt"))
  quit(status = 1)
}

s  <- summary(fit)
co <- data.frame(term = rownames(s), estimate = s[, "theta"],
                 std_error = s[, "se"], p_value = s[, "pvalue"])
write.csv(co, paste0(out_prefix, "_coef.csv"), row.names = FALSE)

set.seed(2)
sims <- simulate(fit, nsim = nsim)
rows <- vector("list", nsim)
for (i in seq_len(nsim)) {
  A <- as.matrix(as.network(sims[[i]]))
  idx <- which(A == 1, arr.ind = TRUE)
  rows[[i]] <- data.frame(sim = i - 1L, from = idx[, 1] - 1L, to = idx[, 2] - 1L)
}
write.csv(do.call(rbind, rows), paste0(out_prefix, "_simedges.csv"), row.names = FALSE)
writeLines(c(status, paste("fit seconds:", secs), paste("nsim:", nsim)),
           paste0(out_prefix, "_status.txt"))
cat("DONE lolog stars |", status, "| fit", secs, "s | nsim", nsim, "\n")
