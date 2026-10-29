# Tapered ERGM experiment driven entirely by a JSON configuration.
#
# Usage:
#   Rscript scripts/tapered_run.R configs/tapered.json

LIB <- Sys.getenv("R_TAPERED_LIB", unset = "")

if (nzchar(LIB)) {
  if (!dir.exists(LIB)) {
    stop("R_TAPERED_LIB does not exist: ", LIB)
  }

  .libPaths(c(normalizePath(LIB), .libPaths()))
}

suppressMessages({
  library(jsonlite)
  library(network)
  library(ergm)
  library(ergm.tapered)
})


# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------

file_arg <- sub(
  "^--file=",
  "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)
)

ROOT <- normalizePath(
  file.path(dirname(file_arg), ".."),
  winslash = "/"
)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 1) {
  stop("usage: Rscript scripts/tapered_run.R <config.json>")
}

config_file <- normalizePath(
  args[1],
  winslash = "/",
  mustWork = TRUE
)

cfg <- jsonlite::fromJSON(
  config_file,
  simplifyVector = FALSE
)

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}


resolve_path <- function(p) {
  if (is.null(p)) return(NULL)

  # Unix absolute path or Windows drive path.
  if (
    grepl("^/", p) ||
    grepl("^[A-Za-z]:[/\\\\]", p)
  ) {
    return(p)
  }

  file.path(ROOT, p)
}


drop_nulls <- function(x) {
  x[!vapply(x, is.null, logical(1))]
}


# Convert formula-valued arguments stored as strings in JSON.
prepare_fit_args <- function(x, env) {
  x <- drop_nulls(x)

  for (nm in c("constraints", "reference")) {
    if (!is.null(x[[nm]]) && is.character(x[[nm]])) {
      x[[nm]] <- as.formula(x[[nm]], env = env)
    }
  }

  x
}


# ---------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------

spec <- as.integer(cfg$specification)

if (!spec %in% 1:3) {
  stop("specification must be 1, 2, or 3")
}

spec_cfg <- cfg$specifications[[as.character(spec)]]

if (is.null(spec_cfg)) {
  stop("no configuration found for specification ", spec)
}

n_seeds <- as.integer(cfg$n_seeds)

if (is.na(n_seeds) || n_seeds < 1) {
  stop("n_seeds must be a positive integer")
}

seeds <- seq_len(n_seeds)

grid <- unlist(cfg$r_grid, use.names = FALSE)

if (!length(grid)) {
  stop("r_grid cannot be empty")
}

sim_cfg <- cfg$simulation %||% list()

nsim <- as.integer(sim_cfg$nsim %||% 200)
seed_offset <- as.integer(sim_cfg$seed_offset %||% 1000)

out_root <- resolve_path(cfg$out_dir %||% "output/tapered")
out_dir <- file.path(out_root, paste0("spec", spec))

dir.create(
  out_dir,
  showWarnings = FALSE,
  recursive = TRUE
)


# ---------------------------------------------------------------------
# Network and model
# ---------------------------------------------------------------------

files <- spec_cfg$files

ed <- read.csv(
  resolve_path(files$edges)
)

nodes <- NULL

if (!is.null(files$nodes)) {
  nodes <- read.csv(
    resolve_path(files$nodes)
  )
  n_vertices <- nrow(nodes)
} else {
  n_vertices <- as.integer(spec_cfg$n_vertices)
}

net <- network.initialize(
  n_vertices,
  directed = isTRUE(spec_cfg$directed)
)

from_col <- spec_cfg$from_column %||% "from"
to_col   <- spec_cfg$to_column %||% "to"

offset <- if (isTRUE(spec_cfg$zero_based %||% TRUE)) 1L else 0L

edge_matrix <- as.matrix(
  ed[, c(from_col, to_col)]
)

net[edge_matrix + offset] <- 1


# Node attributes.
if (!is.null(spec_cfg$node_attributes)) {
  if (is.null(nodes)) {
    stop("node_attributes specified but no nodes file was provided")
  }

  for (attr_name in names(spec_cfg$node_attributes)) {
    column <- spec_cfg$node_attributes[[attr_name]]
    net %v% attr_name <- as.character(nodes[[column]])
  }
}


# Environment in which the ERGM formula is evaluated.
model_env <- new.env(parent = globalenv())
assign("net", net, envir = model_env)


# Dyadic covariate matrices.
if (!is.null(spec_cfg$edgecov)) {
  for (var_name in names(spec_cfg$edgecov)) {

    mat <- as.matrix(
      read.csv(
        resolve_path(spec_cfg$edgecov[[var_name]]),
        header = FALSE
      )
    )

    assign(
      var_name,
      mat,
      envir = model_env
    )
  }
}


model_cfg <- spec_cfg$model

form <- as.formula(
  paste("net ~", model_cfg$terms),
  env = model_env
)

labels <- unlist(
  model_cfg$labels,
  use.names = FALSE
)


# The script is an estimation experiment.
learn_thetas <- model_cfg$learn_thetas %||% TRUE

if (!isTRUE(learn_thetas)) {
  stop(
    "learn_thetas = false is not supported by this fitting script"
  )
}


# ---------------------------------------------------------------------
# Common estimation/control arguments
# ---------------------------------------------------------------------

estimation_args <- cfg$estimation %||% list()

# r is controlled by r_grid, so ignore a single r in estimation.
estimation_args$r <- NULL

control_args <- cfg$control %||% list()


# Optional theta values are used as starting values.
if (!is.null(model_cfg$thetas)) {

  theta0 <- as.numeric(
    unlist(model_cfg$thetas, use.names = FALSE)
  )

  if (length(theta0) != length(labels)) {
    stop(
      "model.thetas has length ",
      length(theta0),
      " but the specification has ",
      length(labels),
      " statistics"
    )
  }

  if (is.null(control_args$init)) {
    control_args$init <- theta0
  }
}


sim_control_args <- sim_cfg$control %||% list()


cat(
  "Spec", spec,
  "| ergm", as.character(packageVersion("ergm")),
  "| ergm.tapered", as.character(packageVersion("ergm.tapered")),
  "| grid:", paste(grid, collapse = ", "),
  "| seeds:", paste(seeds, collapse = ", "),
  "| nsim:", nsim,
  "\n"
)

cat(
  "observed statistics:",
  round(as.numeric(summary(form)), 3),
  "\n"
)


# ---------------------------------------------------------------------
# Existing results, so interrupted experiments can be resumed
# ---------------------------------------------------------------------

fits_csv <- file.path(
  out_dir,
  "fits.csv"
)

done <- if (file.exists(fits_csv)) {
  read.csv(
    fits_csv,
    stringsAsFactors = FALSE
  )
} else {
  NULL
}

rows <- if (is.null(done)) {
  list()
} else {
  split(done, seq_len(nrow(done)))
}


# ---------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------

for (r_value in grid) {

  lab <- as.character(r_value)

  for (seed in seeds) {

    if (
      !is.null(done) &&
      any(done$label == lab & done$seed == seed)
    ) {
      cat("skip", lab, seed, "(done)\n")
      next
    }

    cat(
      "\n=== Spec", spec,
      "| tapering", lab,
      "| seed", seed,
      "\n"
    )

    status <- "CONVERGED"

    flag <- function(cond) {
      if (
        grepl(
          "converge|degenera|stuck",
          conditionMessage(cond),
          ignore.case = TRUE
        )
      ) {
        status <<- paste(
          "WARNING:",
          conditionMessage(cond)
        )
      }
    }


    # ---------------------------------------------------------------
    # control.ergm.tapered()
    # ---------------------------------------------------------------

    current_control <- control_args
    current_control$seed <- seed

    control_obj <- do.call(
      control.ergm.tapered,
      current_control
    )


    # ---------------------------------------------------------------
    # ergm.tapered()
    # ---------------------------------------------------------------

    current_estimation <- prepare_fit_args(
      estimation_args,
      model_env
    )

    if (tolower(lab) == "auto") {

      # Automatic tapering calibration.
      current_estimation$fixed <- FALSE

    } else {

      current_estimation$r <- as.numeric(lab)

      if (is.null(current_estimation$fixed)) {
        current_estimation$fixed <- TRUE
      }
    }


    fit_args <- c(
      list(form),
      current_estimation,
      list(control = control_obj)
    )


    t0 <- proc.time()[["elapsed"]]

    fit <- withCallingHandlers(

      tryCatch(
        do.call(
          ergm.tapered,
          fit_args
        ),
        error = function(e) {
          status <<- paste(
            "ERROR:",
            conditionMessage(e)
          )
          NULL
        }
      ),

      warning = function(w) {
        flag(w)
        invokeRestart("muffleWarning")
      }
    )


    row <- data.frame(
      label = lab,
      seed = seed,
      status = gsub(
        "[\r\n,]+",
        " ",
        status
      ),
      fit_seconds = round(
        proc.time()[["elapsed"]] - t0,
        1
      ),
      stringsAsFactors = FALSE
    )


    # ---------------------------------------------------------------
    # Save fit and simulate
    # ---------------------------------------------------------------

    if (!is.null(fit)) {

      coef_names <- names(coef(fit))

      keep <- !grepl(
        "Taper_Penalty",
        coef_names
      )

      fit_coef <- coef(fit)[keep]

      s <- summary(fit)$coefficients

      se <- s[
        rownames(s) %in% names(fit_coef),
        2
      ]


      strength <- if (
        is.null(fit$tapering.strength)
      ) {
        NA_real_
      } else {
        fit$tapering.strength
      }

      row$r_used <- if (
        is.null(fit$r)
      ) {
        NA_real_
      } else {
        fit$r
      }

      row$tapering_strength <- strength

      row$tapering_converged <- if (
        is.null(fit$tapering.converged)
      ) {
        NA
      } else {
        fit$tapering.converged
      }

      row$strength_at_bound <-
        tolower(lab) == "auto" &&
        !is.na(strength) &&
        (
          strength > 3 - 1e-3 ||
          strength < 1 / 3 + 1e-3
        )


      row <- cbind(
        row,

        setNames(
          as.list(fit_coef),
          paste0("theta_", labels)
        ),

        setNames(
          as.list(as.numeric(se)),
          paste0("se_", labels)
        ),

        setNames(
          as.list(
            as.numeric(fit$tapering.coef)
          ),
          paste0("tau_", labels)
        )
      )


      # -------------------------------------------------------------
      # Simulation settings also come from JSON.
      # -------------------------------------------------------------

      sim_control_obj <- do.call(
        control.simulate.ergm,
        sim_control_args
      )

      t1 <- proc.time()[["elapsed"]]

      sims <- tryCatch(

        simulate(
          fit,
          nsim = nsim,
          output = "network",
          seed = seed + seed_offset,
          control = sim_control_obj
        ),

        error = function(e) {

          row$status <<- paste(
            row$status,
            "| simulation failed:",
            conditionMessage(e)
          )

          NULL
        }
      )

      row$sim_seconds <- round(
        proc.time()[["elapsed"]] - t1,
        1
      )


      if (!is.null(sims)) {

        edge_lists <- lapply(
          seq_along(sims),
          function(i) {

            e <- as.edgelist(
              sims[[i]]
            )

            if (!nrow(e)) {
              return(NULL)
            }

            data.frame(
              sim = i - 1L,
              from = e[, 1] - 1L,
              to = e[, 2] - 1L
            )
          }
        )

        edge_lists <- Filter(
          Negate(is.null),
          edge_lists
        )

        if (length(edge_lists)) {

          el <- do.call(
            rbind,
            edge_lists
          )

        } else {

          el <- data.frame(
            sim = integer(),
            from = integer(),
            to = integer()
          )
        }

        write.csv(
          el,
          gzfile(
            file.path(
              out_dir,
              sprintf(
                "%s_s%d_simedges.csv.gz",
                lab,
                seed
              )
            )
          ),
          row.names = FALSE
        )
      }
    }


    cat(
      "--- Spec", spec,
      "| tapering", lab,
      "| seed", seed,
      "|", row$status,
      "| fit", row$fit_seconds, "s\n"
    )


    # ---------------------------------------------------------------
    # Update fits.csv immediately so the run is resumable.
    # ---------------------------------------------------------------

    rows[[length(rows) + 1]] <- row

    cols <- unique(
      unlist(
        lapply(rows, names)
      )
    )

    all_rows <- do.call(
      rbind,
      lapply(
        rows,
        function(x) {
          x[setdiff(cols, names(x))] <- NA
          x[cols]
        }
      )
    )

    write.csv(
      all_rows,
      fits_csv,
      row.names = FALSE
    )
  }
}


cat(
  "\nDONE ->",
  out_dir,
  "\n"
)