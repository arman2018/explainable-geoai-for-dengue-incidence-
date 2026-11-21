# GW-RF & G-RF
#packages 
library("readxl")
library("dplyr")
library("tidyr")
library("purrr")
library("tibble")
library("stringr")
library("ranger")
library("ParBayesianOptimization")
library("geosphere")
library("writexl")
library("treeshap")
library("ggplot2")
library("patchwork")
library("scales")
library("sf")
library("RColorBrewer")
library("ggtext")

set.seed(2026)
#data
output <- "C:/Users/hp/OneDrive/Desktop/D/Analysis/Recheck/Analysis"
data_path  <- "C:/Users/hp/OneDrive/Desktop/D/Research_Data/Original_data_code_format.xlsx"
output_dir <- output

N_THREADS <- max(1, parallel::detectCores() - 1)

K_CANDIDATES <- seq(10, 45, by = 5)

BAYES_INIT_POINTS <- 20L
BAYES_ITERS_N     <- 80L

raw <- read_excel(data_path, sheet = 1)
stopifnot(all(c("year", "District", "Latitude", "Longitude", "y") %in% names(raw)))

predictor_vars <- paste0("x", 1:21)
stopifnot(all(predictor_vars %in% names(raw)))
stopifnot(length(predictor_vars) == 21)

df <- raw %>%
  mutate(District = str_trim(District)) %>%
  mutate(District = ifelse(tolower(District) == "cox's bazar", "Cox's Bazar", District)) %>%
  arrange(District, year)

n_districts <- length(unique(df$District))
# DIVISION MAPPING
division_lookup <- c(
  "Dhaka" = "Dhaka", "Faridpur" = "Dhaka", "Gazipur" = "Dhaka", "Gopalganj" = "Dhaka",
  "Kishoreganj" = "Dhaka", "Madaripur" = "Dhaka", "Manikganj" = "Dhaka",
  "Munshiganj" = "Dhaka", "Narayanganj" = "Dhaka", "Narsingdi" = "Dhaka",
  "Rajbari" = "Dhaka", "Shariatpur" = "Dhaka", "Tangail" = "Dhaka",
  "Bandarban" = "Chittagong", "Brahamanbaria" = "Chittagong", "Chandpur" = "Chittagong",
  "Chittagong" = "Chittagong", "Comilla" = "Chittagong", "Cox's Bazar" = "Chittagong",
  "Feni" = "Chittagong", "Khagrachhari" = "Chittagong", "Lakshmipur" = "Chittagong",
  "Noakhali" = "Chittagong", "Rangamati" = "Chittagong",
  "Bogra" = "Rajshahi", "Joypurhat" = "Rajshahi", "Naogaon" = "Rajshahi",
  "Natore" = "Rajshahi", "Nawabganj" = "Rajshahi", "Pabna" = "Rajshahi",
  "Rajshahi" = "Rajshahi", "Sirajganj" = "Rajshahi",
  "Bagerhat" = "Khulna", "Chuadanga" = "Khulna", "Jessore" = "Khulna",
  "Jhenaidah" = "Khulna", "Khulna" = "Khulna", "Kushtia" = "Khulna",
  "Magura" = "Khulna", "Meherpur" = "Khulna", "Narail" = "Khulna", "Satkhira" = "Khulna",
  "Barguna" = "Barisal", "Barisal" = "Barisal", "Bhola" = "Barisal",
  "Jhalokati" = "Barisal", "Patuakhali" = "Barisal", "Pirojpur" = "Barisal",
  "Habiganj" = "Sylhet", "Maulvibazar" = "Sylhet", "Sunamganj" = "Sylhet", "Sylhet" = "Sylhet",
  "Dinajpur" = "Rangpur", "Gaibandha" = "Rangpur", "Kurigram" = "Rangpur",
  "Lalmonirhat" = "Rangpur", "Nilphamari" = "Rangpur", "Panchagarh" = "Rangpur",
  "Rangpur" = "Rangpur", "Thakurgaon" = "Rangpur",
  "Jamalpur" = "Mymensingh", "Mymensingh" = "Mymensingh", "Netrakona" = "Mymensingh",
  "Sherpur" = "Mymensingh"
)
df$Division <- division_lookup[df$District]

if (any(is.na(df$Division))) {
  stop("Unmapped District names found -- update division_lookup for: ",
       paste(unique(df$District[is.na(df$Division)]), collapse = ", "))
}

# TRAIN and TEST SPLIT (TRAIN = 2017-2023, TEST = 2024)
train_df <- df %>% filter(year <= 2023) %>% arrange(District, year) %>% as.data.frame()
test_df  <- df %>% filter(year == 2024)  %>% arrange(District, year) %>% as.data.frame()

divisions <- sort(unique(train_df$Division))
stopifnot(length(divisions) == 8)

# METRIC FUNCTIONS
calc_rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
calc_mae  <- function(y, yhat) mean(abs(y - yhat))

calc_r2_oos <- function(y, yhat, baseline_mean) {
  ss_res <- sum((y - yhat)^2)
  ss_tot <- sum((y - baseline_mean)^2)
  1 - ss_res / ss_tot
}

compute_all_metrics <- function(y, yhat, baseline_mean) {
  tibble(
    R2    = calc_r2_oos(y, yhat, baseline_mean),
    RMSE  = calc_rmse(y, yhat),
    MAE   = calc_mae(y, yhat),
    n_obs = length(y)
  )
}

fit_rf <- function(data, params_list, weight = NULL) {
  X <- as.data.frame(data[, predictor_vars])
  ranger(
    x               = X,
    y               = data$y,
    num.trees       = params_list$num.trees,
    mtry            = params_list$mtry,
    min.node.size   = params_list$min.node.size,
    max.depth       = params_list$max.depth,
    sample.fraction = params_list$sample.fraction,
    replace         = TRUE,
    case.weights    = weight,
    num.threads     = N_THREADS,
    verbose         = FALSE,
    seed            = 2026
  )
}

predict_rf <- function(model, newdata) {
  predict(model, data = as.data.frame(newdata[, predictor_vars]))$predictions
}

# SPATIAL CV RUNNER - GLOBAL RANDOM FOREST
# 8-fold leave-one-Division-out CV, performed on TRAIN (2017-2023) only.
run_spatial_cv_global <- function(train_data, divisions, params_list) {
  oof_pred  <- rep(NA_real_, nrow(train_data))
  fold_rmse <- numeric(length(divisions))
  
  for (i in seq_along(divisions)) {
    dv <- divisions[i]
    tr_idx <- which(train_data$Division != dv)
    va_idx <- which(train_data$Division == dv)
    
    model <- fit_rf(train_data[tr_idx, ], params_list)
    preds <- predict_rf(model, train_data[va_idx, ])
    
    oof_pred[va_idx] <- preds
    fold_rmse[i] <- calc_rmse(train_data$y[va_idx], preds)
  }
  
  list(oof_pred = oof_pred, fold_rmse = fold_rmse)
}

# GLOBAL RANDOM FOREST - BAYESIAN HYPERPARAMETER TUNING
run_bayes_opt_robust <- function(FUN, bounds, initPoints, iters.n, iters.k = 1,
                                 acq = "ei", max_retries = 30, base_seed = 2026,
                                 verbose = 1) {
  target_total <- initPoints + iters.n
  save_path <- tempfile(fileext = ".RDS")
  on.exit(unlink(save_path), add = TRUE)
  
  set.seed(base_seed)
  opt_obj <- tryCatch({
    bayesOpt(FUN = FUN, bounds = bounds, initPoints = initPoints,
             iters.n = iters.n, iters.k = iters.k, acq = acq,
             saveFile = save_path, plotProgress = FALSE, verbose = verbose)
  }, error = function(e) {
    if (file.exists(save_path)) readRDS(save_path) else NULL
  })
  
  attempt <- 1
  while (is.null(opt_obj) || nrow(opt_obj$scoreSummary) < target_total) {
    attempt <- attempt + 1
    if (attempt > max_retries) {
      warning(sprintf(
        "Bayesian optimisation stopped after %d resume attempts with %d/%d evaluations completed; proceeding with the best hyperparameters found so far.",
        attempt - 1, if (!is.null(opt_obj)) nrow(opt_obj$scoreSummary) else 0, target_total))
      break
    }
    if (is.null(opt_obj)) stop("Bayesian optimisation failed before any evaluations completed.")
    
    remaining <- target_total - nrow(opt_obj$scoreSummary)
    set.seed(base_seed + attempt)
    opt_obj <- tryCatch({
      addIterations(opt_obj, iters.n = remaining, iters.k = iters.k,
                    saveFile = save_path, plotProgress = FALSE, verbose = verbose)
    }, error = function(e) {
      if (file.exists(save_path)) readRDS(save_path) else opt_obj
    })
  }
  
  opt_obj
}

bayes_bounds <- list(
  num_treesL      = c(100L, 800L),
  mtryL           = c(1L, length(predictor_vars)),
  min_node_sizeL  = c(1L, 20L),
  max_depthL      = c(2L, 10L),
  sample_fraction = c(0.50, 1.00)
)

rf_cv_score <- function(num_treesL, mtryL, min_node_sizeL, max_depthL, sample_fraction) {
  
  params_list <- list(
    num.trees       = round(num_treesL),
    mtry            = round(mtryL),
    min.node.size   = round(min_node_sizeL),
    max.depth       = round(max_depthL),
    sample.fraction = sample_fraction
  )
  
  cv_result <- run_spatial_cv_global(train_df, divisions, params_list)
  mean_rmse <- mean(cv_result$fold_rmse)
  
  list(Score = -mean_rmse, mean_cv_rmse = mean_rmse)
}

bayes_opt_result <- run_bayes_opt_robust(
  FUN         = rf_cv_score,
  bounds      = bayes_bounds,
  initPoints  = BAYES_INIT_POINTS,
  iters.n     = BAYES_ITERS_N,
  iters.k     = 1,
  acq         = "ei",
  base_seed   = 2026,
  verbose     = 1
)

best_pars <- getBestPars(bayes_opt_result)
best_num_trees       <- round(best_pars$num_treesL)
best_mtry            <- round(best_pars$mtryL)
best_min_node_size   <- round(best_pars$min_node_sizeL)
best_max_depth       <- round(best_pars$max_depthL)
best_sample_fraction <- best_pars$sample_fraction

global_params_list <- list(
  num.trees       = best_num_trees,
  mtry            = best_mtry,
  min.node.size   = best_min_node_size,
  max.depth       = best_max_depth,
  sample.fraction = best_sample_fraction
)

print(global_params_list)

# GLOBAL RANDOM FOREST
global_cv <- run_spatial_cv_global(train_df, divisions, global_params_list)

global_fold_metrics <- map_dfr(seq_along(divisions), function(i) {
  dv <- divisions[i]
  va_idx   <- which(train_df$Division == dv)
  tr_idx   <- which(train_df$Division != dv)
  baseline <- mean(train_df$y[tr_idx])
  m <- compute_all_metrics(train_df$y[va_idx], global_cv$oof_pred[va_idx], baseline)
  bind_cols(tibble(Division = dv), m)
})

global_spatial_cv_summary <- global_fold_metrics %>%
  summarise(across(c(R2, RMSE, MAE),
                   list(mean = ~mean(.x), sd = ~sd(.x)),
                   .names = "{.col}_{.fn}")) %>%
  mutate(Model = "Global Random Forest", .before = 1)
#Pooled OOF performance
global_oof_metrics <- compute_all_metrics(train_df$y, global_cv$oof_pred, mean(train_df$y)) %>%
  mutate(Model = "Global Random Forest", .before = 1)

global_final_model <- fit_rf(train_df, global_params_list)
global_test_pred   <- predict_rf(global_final_model, test_df)
global_test_metrics <- compute_all_metrics(test_df$y, global_test_pred, mean(train_df$y)) %>%
  mutate(Model = "Global Random Forest", .before = 1)

global_train_pred <- predict_rf(global_final_model, train_df)
global_train_metrics <- compute_all_metrics(train_df$y, global_train_pred, mean(train_df$y)) %>%
  mutate(Model = "Global Random Forest", .before = 1)

# GW-RF - GEOGRAPHIC WEIGHTING MACHINERY
centroids <- train_df %>% distinct(District, Latitude, Longitude) %>% as.data.frame()
rownames(centroids) <- centroids$District

# Haversine great-circle distance matrix (km)
coords_lonlat <- as.matrix(centroids[, c("Longitude", "Latitude")])
dist_matrix_km <- geosphere::distm(coords_lonlat, coords_lonlat, fun = geosphere::distHaversine) / 1000
rownames(dist_matrix_km) <- centroids$District
colnames(dist_matrix_km) <- centroids$District

# Adaptive bisquare weights
bisquare_weights <- function(target_district, available_districts, k, dist_mat) {
  d <- dist_mat[target_district, available_districts]
  d <- sort(d)
  kk <- min(k, length(d))
  bw <- d[kk]
  if (bw <= 0) {
    positive <- d[d > 0]
    bw <- if (length(positive) > 0) min(positive) else 1
  }
  w <- setNames(rep(0, length(available_districts)), available_districts)
  within_bw <- d[d <= bw]
  w[names(within_bw)] <- (1 - (within_bw / bw)^2)^2
  w
}

fit_local_model <- function(target_district, pool_data, k, dist_mat) {
  available_districts <- unique(pool_data$District)
  w_district <- bisquare_weights(target_district, available_districts, k, dist_mat)
  row_w <- w_district[pool_data$District]
  
  use <- row_w > 0
  X <- as.data.frame(pool_data[use, predictor_vars])
  ranger(
    x               = X,
    y               = pool_data$y[use],
    num.trees       = global_params_list$num.trees,
    mtry            = global_params_list$mtry,
    min.node.size   = global_params_list$min.node.size,
    max.depth       = global_params_list$max.depth,
    sample.fraction = global_params_list$sample.fraction,
    replace         = TRUE,
    case.weights    = row_w[use],
    num.threads     = N_THREADS,
    verbose         = FALSE,
    seed            = 2026
  )
}

fit_predict_local <- function(target_district, pool_data, target_rows, k, dist_mat) {
  model <- fit_local_model(target_district, pool_data, k, dist_mat)
  predict(model, data = as.data.frame(target_rows[, predictor_vars]))$predictions
}

# GW-RF - SPATIAL CV RUNNER
run_spatial_cv_gw <- function(train_data, divisions, k, dist_mat) {
  oof_pred  <- rep(NA_real_, nrow(train_data))
  fold_rmse <- numeric(length(divisions))
  
  for (i in seq_along(divisions)) {
    dv <- divisions[i]
    pool_data <- train_data[train_data$Division != dv, ]
    held_districts <- unique(train_data$District[train_data$Division == dv])
    
    for (target in held_districts) {
      target_idx  <- which(train_data$District == target & train_data$Division == dv)
      target_rows <- train_data[target_idx, ]
      preds <- fit_predict_local(target, pool_data, target_rows, k, dist_mat)
      oof_pred[target_idx] <- preds
    }
    
    va_idx <- which(train_data$Division == dv)
    fold_rmse[i] <- calc_rmse(train_data$y[va_idx], oof_pred[va_idx])
  }
  
  list(oof_pred = oof_pred, fold_rmse = fold_rmse)
}

# GW-RF - BANDWIDTH (k) SELECTION 
k_grid_results <- map_dfr(K_CANDIDATES, function(k) {
  cv_res <- run_spatial_cv_gw(train_df, divisions, k, dist_matrix_km)
  tibble(k = k, mean_cv_rmse = mean(cv_res$fold_rmse), sd_cv_rmse = sd(cv_res$fold_rmse))
})

print(k_grid_results)

best_k <- k_grid_results$k[which.min(k_grid_results$mean_cv_rmse)]

gw_cv <- run_spatial_cv_gw(train_df, divisions, best_k, dist_matrix_km)

gw_fold_metrics <- map_dfr(seq_along(divisions), function(i) {
  dv <- divisions[i]
  va_idx   <- which(train_df$Division == dv)
  tr_idx   <- which(train_df$Division != dv)
  baseline <- mean(train_df$y[tr_idx])
  m <- compute_all_metrics(train_df$y[va_idx], gw_cv$oof_pred[va_idx], baseline)
  bind_cols(tibble(Division = dv), m)
})

gw_spatial_cv_summary <- gw_fold_metrics %>%
  summarise(across(c(R2, RMSE, MAE),
                   list(mean = ~mean(.x), sd = ~sd(.x)),
                   .names = "{.col}_{.fn}")) %>%
  mutate(Model = "GW-RF", .before = 1)

gw_oof_metrics <- compute_all_metrics(train_df$y, gw_cv$oof_pred, mean(train_df$y)) %>%
  mutate(Model = "GW-RF", .before = 1)

all_districts <- unique(train_df$District)
gw_test_pred  <- rep(NA_real_, nrow(test_df))
gw_train_pred <- rep(NA_real_, nrow(train_df))

for (target in all_districts) {
  local_model <- fit_local_model(target, train_df, best_k, dist_matrix_km)
  
  train_idx <- which(train_df$District == target)
  gw_train_pred[train_idx] <- predict(local_model, data = as.data.frame(train_df[train_idx, predictor_vars]))$predictions
  
  test_idx <- which(test_df$District == target)
  if (length(test_idx) > 0) {
    gw_test_pred[test_idx] <- predict(local_model, data = as.data.frame(test_df[test_idx, predictor_vars]))$predictions
  }
}

stopifnot(!anyNA(gw_test_pred))
stopifnot(!anyNA(gw_train_pred))

gw_test_metrics <- compute_all_metrics(test_df$y, gw_test_pred, mean(train_df$y)) %>%
  mutate(Model = "GW-RF", .before = 1)

gw_train_metrics <- compute_all_metrics(train_df$y, gw_train_pred, mean(train_df$y)) %>%
  mutate(Model = "GW-RF", .before = 1)

# FINAL COMPARISON - GLOBAL RANDOM FOREST vs. GW-RF
comparison <- tibble(
  Model         = c("Global Random Forest", "GW-RF"),
  CV_RMSE       = c(mean(global_cv$fold_rmse), mean(gw_cv$fold_rmse)),
  OOF_RMSE      = c(global_oof_metrics$RMSE, gw_oof_metrics$RMSE),
  Test2024_RMSE = c(global_test_metrics$RMSE, gw_test_metrics$RMSE),
  Test2024_MAE  = c(global_test_metrics$MAE, gw_test_metrics$MAE),
  Test2024_R2   = c(global_test_metrics$R2, gw_test_metrics$R2)
)

print(comparison)


global_params_table <- tibble(
  Parameter = c("num_trees", "mtry", "min_node_size", "max_depth", "sample_fraction"),
  Value = c(best_num_trees, best_mtry, best_min_node_size, best_max_depth, best_sample_fraction)
)
#SAVE ACCURACY RESULTS & MODEL PARAMETERS 
gw_params_table <- bind_rows(
  global_params_table,
  tibble(Parameter = "bandwidth_k_nearest_neighbours", Value = best_k)
) %>%
  mutate(Note = ifelse(Parameter == "bandwidth_k_nearest_neighbours",
                       "Selected via spatial-CV grid search",
                       "Inherited from tuned Global Random Forest"))

write_xlsx(
  list(
    Model_Comparison             = comparison,
    Global_Hyperparameters       = global_params_table,
    GW_Hyperparameters           = gw_params_table,
    Bandwidth_Grid_Search        = k_grid_results,
    Global_Train_2017_2023       = global_train_metrics,
    GW_Train_2017_2023           = gw_train_metrics,
    Global_SpatialCV_byDivision  = global_fold_metrics,
    Global_SpatialCV_Summary     = global_spatial_cv_summary,
    Global_OOF_2017_2023         = global_oof_metrics,
    Global_Test_2024             = global_test_metrics,
    GW_SpatialCV_byDivision      = gw_fold_metrics,
    GW_SpatialCV_Summary         = gw_spatial_cv_summary,
    GW_OOF_2017_2023             = gw_oof_metrics,
    GW_Test_2024                 = gw_test_metrics
  ),
  path = file.path(output_dir, "Dengue_RF_vs_GWRF_Accuracy_Results.xlsx")
)

# TRAIN: Observed, Predicted, Residual
All_train_predictions <- train_df %>%
  transmute(
    Year = year,
    District = District,
    Observed = y,
    `Global RF_Pred` = global_train_pred,
    `GW-RF_Pred` = gw_train_pred,
    `Global RF_Resi` = Observed - `Global RF_Pred`,
    `GW-RF_Resi` = Observed - `GW-RF_Pred`
  )

# TRAIN: Observed, Predicted, Residual
All_test_predictions <- test_df %>%
  transmute(
    Year = year,
    District = District,
    Observed = y,
    `Global RF_Pred` = global_test_pred,
    `GW-RF_Pred` = gw_test_pred,
    `Global RF_Resi` = Observed - `Global RF_Pred`,
    `GW-RF_Resi` = Observed - `GW-RF_Pred`
  )

out_dir <- output

write_xlsx(
  list(
    All_train_predictions = All_train_predictions,
    All_test_predictions  = All_test_predictions
  ),
  path = file.path(out_dir, "RF_obs_pre_residual.xlsx")
)

# SHAP EXPLAINABILITY

shap_dir <- output

save_plot_png_tiff <- function(path_no_ext, plot, width, height, dpi = 300) {
  ggsave(paste0(path_no_ext, ".png"),  plot, width = width, height = height, dpi = dpi)
  ggsave(paste0(path_no_ext, ".tiff"), plot, width = width, height = height, dpi = dpi,
         compression = "lzw")
}

predictor_labels <- c(
  x1  = "PREP",             x2  = "RH2M",         x3  = "T2M",
  x4  = "Max_T2M",          x5  = "Min_T2M",      x6  = "WS10M_MAX",
  x7  = "WS10M_MIN",        x8  = "Surface_Pressure", x9  = "POP_DEN",
  x10 = "POP_GR",           x11 = "PM2.5",        x12 = "NO2",
  x13 = "CO",               x14 = "Ozone",        x15 = "SO2",
  x16 = "Forest",           x17 = "PWET",         x18 = "Croplands",
  x19 = "URB_BUILTUP",      x20 = "LST",          x21 = "Elevation"
)
stopifnot(all(predictor_vars %in% names(predictor_labels)))

compute_local_shap <- function(target_district, pool_data, target_rows, k, dist_mat) {
  available_districts <- unique(pool_data$District)
  w_district <- bisquare_weights(target_district, available_districts, k, dist_mat)
  row_w <- w_district[pool_data$District]
  use <- row_w > 0
  ref_data <- as.data.frame(pool_data[use, predictor_vars])
  
  local_model <- fit_local_model(target_district, pool_data, k, dist_mat)
  unified <- treeshap::ranger.unify(local_model, ref_data)
  treeshap::treeshap(unified, as.data.frame(target_rows[, predictor_vars]), verbose = FALSE)$shaps
}

shap_rows <- list(); obs_rows <- list(); district_id <- character(0)

for (target in all_districts) {
  test_idx <- which(test_df$District == target)
  if (length(test_idx) == 0) next
  target_rows <- test_df[test_idx, ]
  shap_rows[[length(shap_rows) + 1]] <- compute_local_shap(target, train_df, target_rows, best_k, dist_matrix_km)
  obs_rows[[length(obs_rows) + 1]]   <- target_rows[, predictor_vars]
  district_id <- c(district_id, rep(target, nrow(target_rows)))
}

shap_df <- do.call(rbind, shap_rows); rownames(shap_df) <- NULL
obs_df  <- do.call(rbind, obs_rows);  rownames(obs_df)  <- NULL
stopifnot(nrow(shap_df) == nrow(test_df), nrow(obs_df) == nrow(test_df))

# PANEL A- SHAP FEATURE IMPORTANCE & PANEL B- SHAP SUMMARY
mean_abs_shap <- colMeans(abs(shap_df))
feature_order <- names(sort(mean_abs_shap, decreasing = TRUE))
feature_levels_display <- rev(predictor_labels[feature_order])

imp_df <- tibble(
  feature    = factor(predictor_labels[feature_order], levels = feature_levels_display),
  importance = mean_abs_shap[feature_order]
)

plot_shap_importance <- ggplot(imp_df, aes(x = importance, y = feature, fill = importance)) +
  geom_col(width = 0.7) +
  scale_fill_gradientn(colors = c("#009E9E", "#7B2FBE", "#D6006E"), guide = "none") +
  labs(x = "mean(|SHAP value|)", y = NULL, title = "A) SHAP feature importance") +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.y = element_text(face = "bold", size = 13, color = "black"),
    axis.title.x = element_text(face = "bold", size = 14, color = "black"),
    plot.title = element_text(face = "bold", size = 16, color = "black"),
    axis.line.x = element_line(color = "black", linewidth = 0.5),
    axis.line.y = element_line(color = "black", linewidth = 0.5),
    axis.ticks = element_line(color = "black", linewidth = 0.4),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank()
  )

PLOT_B_HEIGHT_IN <- 7
PLOT_B_PANEL_HEIGHT_IN <- PLOT_B_HEIGHT_IN - 1.3

shap_long <- as.data.frame(shap_df) %>%
  mutate(.obs = row_number()) %>%
  pivot_longer(-.obs, names_to = "feature", values_to = "shap_value")

obs_long <- as.data.frame(obs_df) %>%
  mutate(.obs = row_number()) %>%
  pivot_longer(-.obs, names_to = "feature", values_to = "raw_value") %>%
  group_by(feature) %>%
  mutate(value_scaled = if (diff(range(raw_value)) > 0)
    (raw_value - min(raw_value)) / (max(raw_value) - min(raw_value)) else 0.5) %>%
  ungroup()

summary_df <- shap_long %>%
  left_join(obs_long, by = c(".obs", "feature")) %>%
  mutate(feature = factor(predictor_labels[feature], levels = feature_levels_display))

set.seed(2026)
plot_shap_summary <- ggplot(summary_df, aes(x = shap_value, y = feature, color = value_scaled)) +
  geom_vline(xintercept = 0, color = "black", linewidth = 0.6) +
  geom_jitter(height = 0.32, width = 0, size = 1.6, alpha = 0.85) +
  scale_color_gradientn(
    colors = c("#2222D4", "#8A2BE2", "#D6006E", "#FF1493"),
    breaks = c(0, 1), labels = c("Low", "High"),
    guide = guide_colorbar(
      title = "Feature value",
      title.position = "right",
      barwidth  = unit(0.32, "cm"),
      barheight = unit(PLOT_B_PANEL_HEIGHT_IN, "in"),
      ticks = FALSE
    )
  ) +
  labs(x = "SHAP value (impact on model output)", y = NULL, title = "B) SHAP summary") +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.title.x = element_text(face = "bold", size = 14, color = "black"),
    plot.title = element_text(face = "bold", size = 16, color = "black"),
    axis.line.x = element_line(color = "black", linewidth = 0.5),
    axis.line.y = element_line(color = "grey55", linewidth = 0.3),
    legend.title = element_text(angle = 90, hjust = 0.5, face = "bold"),
    legend.text = element_text(size = 10),
    panel.grid.minor = element_blank()
  )

plot_shap_AB <- plot_shap_importance + plot_shap_summary + patchwork::plot_layout(widths = c(1, 1.15))

save_plot_png_tiff(
  file.path(shap_dir, "SHAP_Importance_Summary_GWRF_2024"),
  plot_shap_AB, width = 15, height = PLOT_B_HEIGHT_IN, dpi = 300
)

# SHAP DEPENDENCE PLOTS
pick_interaction_partner <- function(target, shap_df, obs_df) {
  x <- obs_df[[target]]; y <- shap_df[[target]]
  other_feats <- setdiff(colnames(obs_df), target)
  sm <- tryCatch(loess(y ~ x, span = 0.9), error = function(e) NULL)
  resid_y <- if (!is.null(sm)) residuals(sm) else y - mean(y)
  cors <- sapply(other_feats, function(k) {
    v <- obs_df[[k]]
    if (sd(v) == 0) return(0)
    suppressWarnings(abs(cor(resid_y, v, use = "complete.obs")))
  })
  other_feats[which.max(cors)]
}

dependence_plot_order <- c("x4", "x10", "x3", "x5", "x6", "x8", "x14", "x19",
                           "x9", "x2", "x11", "x12", "x18", "x20", "x21",
                           "x17", "x7", "x1", "x13", "x16", "x15")
stopifnot(setequal(dependence_plot_order, predictor_vars))

DEP_PANEL_WIDTH_IN       <- 6.7
DEP_PANEL_HEIGHT_IN      <- 3.8
DEP_PANEL_DATA_HEIGHT_IN <- DEP_PANEL_HEIGHT_IN - 1.3

make_dependence_plot <- function(target, panel_label, shap_df, obs_df, labels, smooth_pos = "top") {
  partner <- pick_interaction_partner(target, shap_df, obs_df)
  df <- data.frame(
    feature_value = obs_df[[target]],
    shap_value    = shap_df[[target]],
    partner_value = obs_df[[partner]]
  )
  x_rng <- range(df$feature_value); y_rng <- range(df$shap_value)
  
  label_x <- x_rng[1] + 0.05 * diff(x_rng)
  label_y <- if (smooth_pos == "bottom") y_rng[1] + 0.08 * diff(y_rng) else y_rng[2] - 0.06 * diff(y_rng)
  
  ggplot(df, aes(x = feature_value, y = shap_value)) +
    geom_point(aes(color = partner_value), size = 2.0, alpha = 0.85) +
    geom_smooth(method = "loess", se = FALSE, color = "#1B7A3D",
                linewidth = 1.2, span = 0.9) +
    annotate("segment", x = label_x, xend = label_x + 0.07 * diff(x_rng),
             y = label_y, yend = label_y, color = "#1B7A3D", linewidth = 1.2) +
    annotate("text", x = label_x + 0.09 * diff(x_rng), y = label_y,
             label = "Descriptive smooth", hjust = 0, size = 4.2, fontface = "bold",
             color = "black") +
    scale_color_gradientn(
      colors = c("#1E3FE0", "#B02BD6", "#E8006E"), name = labels[partner],
      guide = guide_colorbar(
        title.position = "right",
        barwidth  = unit(0.25, "cm"),
        barheight = unit(DEP_PANEL_DATA_HEIGHT_IN, "in")
      )
    ) +
    labs(title = paste0("(", panel_label, ") ", labels[target], " Dependence Plot"),
         x = "Feature Value", y = "SHAP Value") +
    theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", size = 17, hjust = 0, color = "black"),
      axis.title = element_text(face = "bold", size = 15, color = "black"),
      axis.text = element_text(size = 13, color = "black"),
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black", linewidth = 0.4),
      legend.title = element_text(angle = 90, hjust = 0.5, size = 14, face = "bold"),
      legend.text = element_text(size = 13, color = "black"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey90", linewidth = 0.25),
      plot.margin = margin(8, 12, 8, 8)
    )
}

panel_labels_21 <- letters[seq_along(dependence_plot_order)]
smooth_pos_21   <- ifelse(dependence_plot_order == "x6", "bottom", "top")

dependence_plots <- Map(
  function(v, lbl, pos) make_dependence_plot(v, lbl, shap_df = shap_df, obs_df = obs_df,
                                             labels = predictor_labels, smooth_pos = pos),
  dependence_plot_order, panel_labels_21, smooth_pos_21
)
names(dependence_plots) <- dependence_plot_order

combined_dependence_plot <- patchwork::wrap_plots(dependence_plots[dependence_plot_order],
                                                  ncol = 3, nrow = 7)

save_plot_png_tiff(
  file.path(shap_dir, "SHAP_Dependence_AllVariables_Combined"),
  combined_dependence_plot,
  width = DEP_PANEL_WIDTH_IN * 3, height = DEP_PANEL_HEIGHT_IN * 7, dpi = 300
)

shap_values_wide <- as.data.frame(shap_df)
colnames(shap_values_wide) <- predictor_labels[colnames(shap_values_wide)]
shap_values_wide <- bind_cols(tibble(District = test_df$District), shap_values_wide)

feature_values_wide <- as.data.frame(obs_df)
colnames(feature_values_wide) <- predictor_labels[colnames(feature_values_wide)]
feature_values_wide <- bind_cols(tibble(District = test_df$District), feature_values_wide)

shap_importance_table <- tibble(
  Variable = predictor_labels[feature_order],
  Mean_Abs_SHAP = mean_abs_shap[feature_order]
)

dependence_data <- map_dfr(dependence_plot_order, function(v) {
  partner <- pick_interaction_partner(v, shap_df, obs_df)
  tibble(
    District                   = test_df$District,
    Variable                   = predictor_labels[v],
    Feature_Value              = obs_df[[v]],
    SHAP_Value                 = shap_df[[v]],
    Interaction_Partner        = predictor_labels[partner],
    Interaction_Partner_Value  = obs_df[[partner]]
  )
})

write_xlsx(
  list(
    SHAP_Values_2024        = shap_values_wide,
    Feature_Values_2024     = feature_values_wide,
    SHAP_Feature_Importance = shap_importance_table,
    SHAP_Dependence_Data    = dependence_data
  ),
  path = file.path(shap_dir, "Dengue_GWRF_SHAP_Values.xlsx")
)

# SPATIAL DISTRIBUTION OF |SHAP VALUE| 
district_abs_shap <- as.data.frame(shap_df)[, predictor_vars] %>%
  abs() %>%
  bind_cols(tibble(District = test_df$District), .) %>%
  group_by(District) %>%
  summarise(across(all_of(predictor_vars), mean, na.rm = TRUE), .groups = "drop")

stopifnot(nrow(district_abs_shap) == length(unique(test_df$District)))

shp_path      <- "C:/Users/hp/OneDrive/Desktop/Multi- Diseases/Airborne/Shapefile/gadm41_BGD_2.shp"
district_name <- "NAME_2"

bgd_adm2 <- sf::st_read(shp_path, quiet = TRUE)
stopifnot(district_name %in% names(bgd_adm2))

gadm_to_project_name <- c(
  "Bogura"           = "Bogra",
  "Barishal"         = "Barisal",
  "Cumilla"          = "Comilla",
  "Chattogram"       = "Chittagong",
  "Jashore"          = "Jessore",
  "Chapai Nawabganj" = "Nawabganj",
  "Brahmanbaria"     = "Brahamanbaria",
  "Netrokona"        = "Netrakona",
  "Moulvibazar"      = "Maulvibazar",
  "Maulvi Bazar"     = "Maulvibazar",
  "Jhenaidaha"       = "Jhenaidah",
  "Coxs Bazar"       = "Cox's Bazar",
  "Cox'S Bazar"      = "Cox's Bazar"
)

bgd_adm2 <- bgd_adm2 %>%
  mutate(District_clean = str_trim(.data[[district_name]])) %>%
  mutate(District_clean = ifelse(District_clean %in% names(gadm_to_project_name),
                                 gadm_to_project_name[District_clean],
                                 District_clean))

shap_map_sf <- bgd_adm2 %>%
  select(District_clean, geometry) %>%
  rename(District = District_clean) %>%
  inner_join(district_abs_shap, by = "District")

stopifnot(nrow(shap_map_sf) == nrow(district_abs_shap))

shap_color_palette <- rev(RColorBrewer::brewer.pal(11, "RdBu"))

make_shap_spatial_map <- function(var_x, panel_letter, map_data,
                                  label = predictor_labels[[var_x]],
                                  colors_low_high = shap_color_palette) {
  
  rng <- range(map_data[[var_x]], na.rm = TRUE)
  
  ggplot(map_data) +
    geom_sf(aes(fill = .data[[var_x]]), color = "white", linewidth = 0.2) +
    scale_fill_gradientn(
      colors = colors_low_high,
      limits = rng,
      breaks = rng,
      labels = c(sprintf("Low: %.2f", rng[1]), sprintf("High: %.2f", rng[2])),
      guide  = guide_colorbar(
        title          = "SHAP value",
        title.position = "top",
        barwidth       = unit(0.4, "cm"),
        barheight      = unit(3.2, "cm"),
        ticks          = FALSE
      )
    ) +
    labs(title = paste0(panel_letter, ") ", label)) +
    theme_void(base_size = 14) +
    theme(
      plot.title      = element_text(face = "bold", size = 20, hjust = 0.5,
                                     color = "black", margin = margin(b = 8)),
      legend.title    = element_text(angle = 0, hjust = 0.5, size = 14,
                                     face = "bold", color = "black",
                                     margin = margin(b = 16)),
      legend.text     = element_text(size = 12, face = "bold",
                                     color = "black", hjust = 0.5),
      legend.position = "right",
      legend.box.margin = margin(0, 0, 0, 4),
      plot.margin     = margin(12, 18, 12, 12)
    )
}

all_map_plots <- Map(
  function(v, lbl) make_shap_spatial_map(v, lbl, shap_map_sf),
  dependence_plot_order, panel_labels_21
)

n_map_cols <- 4
n_map_rows <- ceiling(length(all_map_plots) / n_map_cols)

combined_map_fig <- patchwork::wrap_plots(all_map_plots,
                                          ncol = n_map_cols,
                                          nrow = n_map_rows)

fig_width_in  <- 20
fig_height_in <- 29
fig_dpi       <- 750

save_plot_png_tiff(
  file.path(shap_dir, "SHAP_Spatial_Distribution_Combined"),
  combined_map_fig, width = fig_width_in, height = fig_height_in, dpi = fig_dpi
)

district_abs_shap_named <- district_abs_shap
names(district_abs_shap_named)[-1] <- predictor_labels[names(district_abs_shap_named)[-1]]

writexl::write_xlsx(
  list(
    District_AbsSHAP_2024_x_codes = district_abs_shap,
    District_AbsSHAP_2024_named   = district_abs_shap_named
  ),
  path = file.path(shap_dir, "Dengue_GWRF_SHAP_Spatial_Distribution.xlsx")
)

# VARIABLE-LEVEL SHAP EFFECT & IMPORTANCE
CAT_BAR_COLOR <- "#C0392B"

n_test_obs <- nrow(shap_df)

var_effect_summary <- purrr::map_dfr(predictor_vars, function(v) {
  vals  <- shap_df[[v]]
  m     <- mean(vals)
  se    <- sd(vals) / sqrt(n_test_obs)
  tcrit <- qt(0.975, df = n_test_obs - 1)
  tibble(
    x_code        = v,
    Variable      = predictor_labels[[v]],
    n             = n_test_obs,
    mean_shap     = m,
    ci_lower      = m - tcrit * se,
    ci_upper      = m + tcrit * se,
    mean_abs_shap = mean(abs(vals))
  )
})

var_order <- var_effect_summary %>% dplyr::arrange(mean_abs_shap) %>% dplyr::pull(Variable)
var_effect_summary <- var_effect_summary %>%
  dplyr::mutate(Variable = factor(Variable, levels = var_order))

panelA_var_effect <- ggplot(var_effect_summary, aes(x = mean_shap, y = Variable)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40", linewidth = 0.5) +
  geom_col(fill = CAT_BAR_COLOR, width = 0.65) +
  geom_errorbar(aes(xmin = ci_lower, xmax = ci_upper),
                width = 0.22, linewidth = 0.6, color = "black") +
  geom_text(aes(x = pmin(mean_shap, ci_lower), label = paste0("n=", n)),
            hjust = 1.15, size = 3.6, color = "black") +
  scale_x_continuous(expand = expansion(mult = c(0.18, 0.06))) +
  labs(x = "Mean SHAP (prediction scale)", y = NULL,
       title = "a) Variable effect (mean SHAP \u00b1 95% CI)") +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.y         = element_text(face = "bold", size = 12, color = "black"),
    axis.title.x        = element_text(face = "bold", size = 13, color = "black"),
    plot.title          = element_text(face = "bold", size = 15, color = "black"),
    axis.line           = element_line(color = "black", linewidth = 0.5),
    axis.ticks          = element_line(color = "black", linewidth = 0.4),
    panel.grid.major.y  = element_blank(),
    panel.grid.minor    = element_blank()
  )

panelB_var_importance <- ggplot(var_effect_summary, aes(x = mean_abs_shap, y = Variable)) +
  geom_col(fill = CAT_BAR_COLOR, width = 0.65) +
  labs(x = "Mean |SHAP| (magnitude)", y = NULL,
       title = "b) Variable importance (mean |SHAP|)") +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.y         = element_blank(),
    axis.ticks.y        = element_blank(),
    axis.title.x        = element_text(face = "bold", size = 13, color = "black"),
    plot.title          = element_text(face = "bold", size = 15, color = "black"),
    axis.line.x         = element_line(color = "black", linewidth = 0.5),
    axis.line.y         = element_line(color = "grey55", linewidth = 0.3),
    axis.ticks.x        = element_line(color = "black", linewidth = 0.4),
    panel.grid.major.y  = element_blank(),
    panel.grid.minor    = element_blank()
  )

panelAB_var <- panelA_var_effect + panelB_var_importance +
  patchwork::plot_layout(widths = c(1.25, 1))

save_plot_png_tiff(
  file.path(shap_dir, "SHAP_Variable_Level_Effect_Importance_2024"),
  panelAB_var, width = 13, height = 7, dpi = 300
)

writexl::write_xlsx(
  list(
    Variable_Level_SHAP_Effect_Importance = var_effect_summary %>%
      dplyr::select(x_code, Variable, n, mean_shap, ci_lower, ci_upper, mean_abs_shap)
  ),
  path = file.path(shap_dir, "Dengue_GWRF_SHAP_Variable_Level_Effect.xlsx")
)

# SHAP WATERFALL PLOTS
waterfall_output_dir <- output

FIG_DPI <- 750

N_TOP_FEATURES <- 10

COLOR_POS <- "#FF0051"
COLOR_NEG <- "#008BFB"

test_summary <- tibble(
  District  = test_df$District,
  Division  = test_df$Division,
  True_2024 = test_df$y,
  Pred_2024 = gw_test_pred
)

max_rows <- test_summary %>%
  group_by(Division) %>%
  slice_max(Pred_2024, n = 1, with_ties = FALSE) %>%
  ungroup() %>% arrange(Division) %>%
  mutate(Sample_Type = "Max Prediction Sample")

min_rows <- test_summary %>%
  group_by(Division) %>%
  slice_min(Pred_2024, n = 1, with_ties = FALSE) %>%
  ungroup() %>% arrange(Division) %>%
  mutate(Sample_Type = "Min Prediction Sample")

representative_rows <- bind_rows(max_rows, min_rows)

top_row    <- representative_rows %>% filter(Pred_2024 == max(Pred_2024)) %>% slice(1)
bottom_row <- representative_rows %>% filter(Pred_2024 == min(Pred_2024)) %>% slice(1)
extreme_rows   <- bind_rows(top_row, bottom_row)
remaining_rows <- representative_rows %>% filter(!District %in% extreme_rows$District)

get_shap_row <- function(district) {
  idx <- which(district_id == district)
  stopifnot(length(idx) == 1)
  shap_row <- as.numeric(as.data.frame(shap_df)[idx, predictor_vars])
  names(shap_row) <- predictor_vars
  shap_row
}

build_waterfall_data <- function(district, pred_value, true_value, top_n) {
  shap_row   <- get_shap_row(district)
  base_value <- pred_value - sum(shap_row)
  
  ord        <- order(abs(shap_row), decreasing = TRUE)
  top_vars   <- names(shap_row)[ord][seq_len(min(top_n, length(shap_row)))]
  other_vars <- setdiff(names(shap_row), top_vars)
  
  feat_names <- unname(predictor_labels[top_vars])
  feat_vals  <- unname(shap_row[top_vars])
  if (length(other_vars) > 0) {
    feat_names <- c(feat_names, "Others")
    feat_vals  <- c(feat_vals, sum(shap_row[other_vars]))
  }
  
  wf <- tibble(Feature = feat_names, ShapValue = feat_vals) %>%
    mutate(
      step      = row_number(),
      n_steps   = n(),
      y_pos     = n_steps - step + 1,
      cum_end   = base_value + cumsum(ShapValue),
      cum_start = lag(cum_end, default = base_value),
      Direction = factor(ifelse(ShapValue >= 0, "Increases prediction", "Decreases prediction"),
                         levels = c("Increases prediction", "Decreases prediction"))
    )
  
  list(wf = wf, base_value = base_value, pred_value = pred_value, true_value = true_value)
}

make_waterfall_plot <- function(district, division, sample_type, pred_value, true_value, top_n,
                                title_prefix = "") {
  
  wd <- build_waterfall_data(district, pred_value, true_value, top_n)
  wf <- wd$wf
  n_steps <- nrow(wf)
  
  title_md <- paste0(
    "<b style='font-size:23pt;color:#000000;'>", title_prefix, sample_type, ": Pred: ",
    sprintf("%.2f", pred_value), " | True: ", sprintf("%.2f", true_value), "</b><br>",
    "<b style='font-size:21pt;color:#0D1B2A;'>District: ", district,
    " | Division: ", division, "</b>"
  )
  
  x_range  <- range(c(wf$cum_start, wf$cum_end, wd$base_value, wd$pred_value))
  x_pad    <- diff(x_range) * 0.04
  label_x  <- wf$cum_end + ifelse(wf$ShapValue >= 0, 1, -1) * (x_pad * 0.45)
  label_hj <- ifelse(wf$ShapValue >= 0, 0, 1)
  
  p <- ggplot(wf) +
    geom_segment(
      data = wf %>% filter(step < n_steps),
      aes(x = cum_end, xend = cum_end, y = y_pos - 0.35, yend = y_pos - 1 + 0.35),
      linetype = "dashed", color = "grey70", linewidth = 0.5
    ) +
    geom_rect(
      aes(xmin = cum_start, xmax = cum_end,
          ymin = y_pos - 0.35, ymax = y_pos + 0.35, fill = Direction),
      color = "grey20", linewidth = 0.3
    ) +
    geom_text(
      aes(x = label_x, y = y_pos, label = sprintf("%+.2f", ShapValue), hjust = label_hj),
      fontface = "bold", size = 6.6, color = "#000000"
    ) +
    geom_vline(xintercept = wd$pred_value, linetype = "dotted",
               color = "black", linewidth = 0.9) +
    annotate("text", x = wd$pred_value, y = n_steps + 0.9,
             label = paste0("f(x) = ", sprintf("%.2f", wd$pred_value)),
             fontface = "bold", size = 5.8, color = "black") +
    scale_fill_manual(values = c("Increases prediction" = COLOR_POS,
                                 "Decreases prediction" = COLOR_NEG),
                      limits = c("Increases prediction", "Decreases prediction"),
                      drop = FALSE, name = NULL) +
    scale_y_continuous(breaks = wf$y_pos, labels = wf$Feature,
                       limits = c(-0.2, n_steps + 1.6), expand = c(0, 0)) +
    scale_x_continuous(labels = scales::comma, expand = expansion(mult = c(0.13, 0.13))) +
    labs(title = title_md, x = NULL, y = NULL) +
    theme_minimal(base_size = 15) +
    theme(
      plot.title         = ggtext::element_markdown(hjust = 0.5, lineheight = 1.25,
                                                    margin = margin(b = 10)),
      axis.text.y        = element_text(face = "bold", size = 21, color = "black"),
      axis.text.x        = element_text(face = "bold", size = 22, color = "black"),
      axis.title.x       = element_text(face = "bold", size = 14, color = "black"),
      axis.line          = element_line(color = "black", linewidth = 0.6),
      axis.ticks         = element_line(color = "black", linewidth = 0.4),
      panel.grid.major.y = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_line(color = "grey90", linewidth = 0.3),
      panel.border       = element_blank(),
      legend.position    = "none",
      plot.margin        = margin(10, 18, 10, 10)
    )
  
  list(plot = p, wf = wf, base_value = wd$base_value, pred_value = wd$pred_value,
       true_value = wd$true_value, district = district, division = division,
       sample_type = sample_type)
}

extreme_prefix <- ifelse(extreme_rows$Sample_Type == "Max Prediction Sample", "(A) ", "(B) ")

extreme_results <- pmap(
  list(extreme_rows$District, extreme_rows$Division, extreme_rows$Sample_Type,
       extreme_rows$Pred_2024, extreme_rows$True_2024, extreme_prefix),
  function(d, dv, s, p, t, pre) make_waterfall_plot(d, dv, s, p, t,
                                                    top_n = length(predictor_vars),
                                                    title_prefix = pre)
)
names(extreme_results) <- paste(extreme_rows$District, extreme_rows$Sample_Type, sep = "__")

extreme_height_in <- 6.5 + (length(predictor_vars) + 1) * 0.34
extreme_width_in  <- 12.5

combined_page1 <- patchwork::wrap_plots(map(extreme_results, "plot"), ncol = 2, nrow = 1)

save_plot_png_tiff(
  file.path(waterfall_output_dir, "Dengue_GWRF_SHAP_Waterfall_Page1_HighestLowest"),
  combined_page1, width = 2 * extreme_width_in, height = extreme_height_in, dpi = FIG_DPI
)

grid_results <- pmap(
  list(remaining_rows$District, remaining_rows$Division, remaining_rows$Sample_Type,
       remaining_rows$Pred_2024, remaining_rows$True_2024),
  function(d, dv, s, p, t) make_waterfall_plot(d, dv, s, p, t, top_n = N_TOP_FEATURES)
)
names(grid_results) <- paste(remaining_rows$District, remaining_rows$Sample_Type, sep = "__")

n_col <- 4
n_row <- ceiling(length(grid_results) / n_col)
panel_height_in <- 6.1 + (N_TOP_FEATURES + 1) * 0.32
panel_width_in  <- 10.2

combined_page2 <- patchwork::wrap_plots(map(grid_results, "plot"), ncol = n_col, nrow = n_row)

save_plot_png_tiff(
  file.path(waterfall_output_dir, "Dengue_GWRF_SHAP_Waterfall_Page2_Remaining14"),
  combined_page2, width = n_col * panel_width_in, height = n_row * panel_height_in, dpi = FIG_DPI
)

waterfall_results <- c(extreme_results, grid_results)

waterfall_table <- imap_dfr(waterfall_results, function(res, key) {
  res$wf %>%
    transmute(District = res$district, Division = res$division, Sample_Type = res$sample_type,
              Feature, SHAP_Value = ShapValue, Cumulative_Start = cum_start,
              Cumulative_End = cum_end) %>%
    bind_rows(tibble(District = res$district, Division = res$division, Sample_Type = res$sample_type,
                     Feature = "Base value", SHAP_Value = NA_real_,
                     Cumulative_Start = NA_real_, Cumulative_End = res$base_value)) %>%
    bind_rows(tibble(District = res$district, Division = res$division, Sample_Type = res$sample_type,
                     Feature = "f(x) [final prediction]", SHAP_Value = NA_real_,
                     Cumulative_Start = NA_real_, Cumulative_End = res$pred_value))
})

waterfall_summary_table <- tibble(
  District    = map_chr(waterfall_results, "district"),
  Division    = map_chr(waterfall_results, "division"),
  Sample_Type = map_chr(waterfall_results, "sample_type"),
  Base_Value  = map_dbl(waterfall_results, "base_value"),
  Pred_2024   = map_dbl(waterfall_results, "pred_value"),
  True_2024   = map_dbl(waterfall_results, "true_value")
)

writexl::write_xlsx(
  list(Waterfall_Summary = waterfall_summary_table, Waterfall_Cascade_Detail = waterfall_table),
  path = file.path(waterfall_output_dir, "Dengue_GWRF_SHAP_Waterfall_Data.xlsx")
)

# PREDICTED DENGUE INCIDENCE & SPATIALLY WEIGHTED EMPIRICAL

uq_output_dir <- output

uq_weighted_quantile <- function(values, weights, probs) {
  ok <- is.finite(values) & is.finite(weights) & weights >= 0
  values <- values[ok]; weights <- weights[ok]
  if (length(values) == 0 || sum(weights) <= 0) return(rep(NA_real_, length(probs)))
  ord <- order(values)
  v <- values[ord]; w <- weights[ord]
  cw <- (cumsum(w) - 0.5 * w) / sum(w)
  stats::approx(x = cw, y = v, xout = probs, rule = 2)$y
}

uq_oof_df <- data.frame(
  District = train_df$District,
  Division = train_df$Division,
  Year     = train_df$year,
  Observed = train_df$y,
  OOF_Pred = gw_cv$oof_pred,
  stringsAsFactors = FALSE
)
uq_oof_df$Residual <- uq_oof_df$Observed - uq_oof_df$OOF_Pred
uq_oof_df <- uq_oof_df[is.finite(uq_oof_df$Residual), ]

uq_pooled_residuals <- uq_oof_df$Residual
uq_available_districts <- rownames(dist_matrix_km)
uq_n <- nrow(test_df)

uq_lower       <- numeric(uq_n)
uq_upper       <- numeric(uq_n)
uq_width       <- numeric(uq_n)
uq_ess         <- numeric(uq_n)
uq_fallback    <- logical(uq_n)
uq_sens_pct    <- numeric(uq_n)

for (i in seq_len(uq_n)) {
  
  target   <- test_df$District[i]
  pred_val <- gw_test_pred[i]
  
  w_unique <- bisquare_weights(target, uq_available_districts, best_k, dist_matrix_km)
  
  ess_denom  <- sum(w_unique^2)
  uq_ess[i]  <- if (ess_denom > 0) (sum(w_unique)^2) / ess_denom else NA_real_
  
  districts_in_window <- names(w_unique)[w_unique > 0]
  
  use_fallback <- FALSE
  local_values <- NULL
  local_weights <- NULL
  
  if (length(districts_in_window) == 0) {
    use_fallback <- TRUE
  } else {
    local_pool <- uq_oof_df[uq_oof_df$District %in% districts_in_window, ]
    if (nrow(local_pool) < 2) {
      use_fallback <- TRUE
    } else {
      local_pool$Weight <- w_unique[local_pool$District]
      if (sum(local_pool$Weight) <= 0) {
        use_fallback <- TRUE
      } else {
        local_values  <- local_pool$Residual
        local_weights <- local_pool$Weight
      }
    }
  }
  
  if (use_fallback) {
    pool_values  <- uq_pooled_residuals
    pool_weights <- rep(1, length(pool_values))
  } else {
    pool_values  <- local_values
    pool_weights <- local_weights
  }
  
  q_full <- uq_weighted_quantile(pool_values, pool_weights, c(0.025, 0.975))
  
  lower_raw   <- pred_val + q_full[1]
  upper_raw   <- pred_val + q_full[2]
  lower_final <- max(lower_raw, 0)
  upper_final <- max(upper_raw, lower_final)
  width_full  <- upper_final - lower_final
  
  if (length(pool_values) > 2) {
    max_idx <- which.max(abs(pool_values))
    v_excl  <- pool_values[-max_idx]
    w_excl  <- pool_weights[-max_idx]
    q_excl  <- uq_weighted_quantile(v_excl, w_excl, c(0.025, 0.975))
    lower_excl <- max(pred_val + q_excl[1], 0)
    upper_excl <- max(pred_val + q_excl[2], lower_excl)
    width_excl <- upper_excl - lower_excl
    sens_pct <- if (width_full > 0) (width_excl - width_full) / width_full * 100 else NA_real_
  } else {
    sens_pct <- NA_real_
  }
  
  uq_lower[i]    <- lower_final
  uq_upper[i]    <- upper_final
  uq_width[i]    <- width_full
  uq_fallback[i] <- use_fallback
  uq_sens_pct[i] <- sens_pct
}

uq_district_results <- data.frame(
  District = test_df$District,
  Division = test_df$Division,
  Observed_2024 = test_df$y,
  Predicted_2024 = gw_test_pred,
  Empirical_Uncertainty_Lower = uq_lower,
  Empirical_Uncertainty_Upper = uq_upper,
  Spatial_Weighted_Empirical_Uncertainty_Width = uq_width,
  Relative_Uncertainty_Width = uq_width / pmax(gw_test_pred, 1),
  Spatial_ESS = uq_ess,
  Covered = (test_df$y >= uq_lower) & (test_df$y <= uq_upper),
  Fallback_Used = uq_fallback,
  Uncertainty_Width_Sensitivity_To_Max_Residual_Pct = uq_sens_pct,
  stringsAsFactors = FALSE
)

stopifnot(nrow(uq_district_results) == uq_n)

uq_n_districts   <- nrow(uq_district_results)
uq_n_covered     <- sum(uq_district_results$Covered)
uq_n_not_covered <- uq_n_districts - uq_n_covered
uq_coverage      <- uq_n_covered / uq_n_districts

uq_mean_width       <- mean(uq_district_results$Spatial_Weighted_Empirical_Uncertainty_Width)
uq_median_width     <- median(uq_district_results$Spatial_Weighted_Empirical_Uncertainty_Width)
uq_min_width        <- min(uq_district_results$Spatial_Weighted_Empirical_Uncertainty_Width)
uq_max_width        <- max(uq_district_results$Spatial_Weighted_Empirical_Uncertainty_Width)
uq_mean_rel_width   <- mean(uq_district_results$Relative_Uncertainty_Width)
uq_median_rel_width <- median(uq_district_results$Relative_Uncertainty_Width)
uq_min_ess          <- min(uq_district_results$Spatial_ESS, na.rm = TRUE)
uq_max_ess          <- max(uq_district_results$Spatial_ESS, na.rm = TRUE)
uq_n_fallback       <- sum(uq_district_results$Fallback_Used)
uq_mean_sens_pct    <- mean(uq_district_results$Uncertainty_Width_Sensitivity_To_Max_Residual_Pct, na.rm = TRUE)

uq_summary <- data.frame(
  Metric = c(
    "N_Districts", "Empirical_Coverage_Proportion", "N_Covered", "N_Not_Covered",
    "Mean_Uncertainty_Width", "Median_Uncertainty_Width",
    "Min_Uncertainty_Width", "Max_Uncertainty_Width",
    "Mean_Relative_Uncertainty_Width", "Median_Relative_Uncertainty_Width",
    "Min_Spatial_ESS", "Max_Spatial_ESS", "N_Fallback_Districts",
    "Mean_Uncertainty_Width_Sensitivity_To_Max_Residual_Pct"
  ),
  Value = c(
    uq_n_districts, uq_coverage, uq_n_covered, uq_n_not_covered,
    uq_mean_width, uq_median_width, uq_min_width, uq_max_width,
    uq_mean_rel_width, uq_median_rel_width,
    uq_min_ess, uq_max_ess, uq_n_fallback, uq_mean_sens_pct
  ),
  stringsAsFactors = FALSE
)

writexl::write_xlsx(
  list(District_Results = uq_district_results, Uncertainty_Summary = uq_summary),
  path = file.path(uq_output_dir, "GW_RF_2024_Prediction_Uncertainty.xlsx")
)

uq_unmatched <- setdiff(uq_district_results$District, bgd_adm2$District_clean)
if (length(uq_unmatched) > 0) {
  stop("Section 25: the following districts did not match a boundary polygon: ",
       paste(uq_unmatched, collapse = ", "))
}

uq_map_sf <- bgd_adm2 %>%
  select(District_clean, geometry) %>%
  rename(District = District_clean) %>%
  inner_join(uq_district_results, by = "District")

stopifnot(nrow(uq_map_sf) == nrow(uq_district_results))

uq_make_map <- function(var, panel_title, legend_title, map_data, fill_colors) {
  rng <- range(map_data[[var]], na.rm = TRUE)
  ggplot(map_data) +
    geom_sf(aes(fill = .data[[var]]), color = "white", linewidth = 0.15) +
    scale_fill_gradientn(
      colors = fill_colors, limits = rng,
      guide = guide_colorbar(
        title          = legend_title,
        title.position = "top",
        title.hjust    = 0.5,
        barwidth       = unit(0.4, "cm"),
        barheight      = unit(5.2, "cm")
      )
    ) +
    labs(title = panel_title) +
    theme_void(base_size = 14) +
    theme(
      plot.title      = element_text(face = "bold", size = 17, hjust = 0, color = "black",
                                     margin = margin(b = 8)),
      legend.title    = element_text(size = 11, face = "bold", color = "black"),
      legend.text     = element_text(size = 9, color = "black"),
      legend.position = "right",
      plot.margin     = margin(6, 14, 6, 6)
    )
}

uq_incidence_palette   <- RColorBrewer::brewer.pal(9, "YlOrRd")
uq_uncertainty_palette <- viridisLite::viridis(9)

uq_map_predicted <- uq_make_map(
  var          = "Predicted_2024",
  panel_title  = "(a) Predicted dengue incidence",
  legend_title = "Predicted Incidence",
  map_data     = uq_map_sf,
  fill_colors  = uq_incidence_palette
)

uq_map_uncertainty <- uq_make_map(
  var          = "Spatial_Weighted_Empirical_Uncertainty_Width",
  panel_title  = "(b) Spatially Weighted Empirical Prediction-Uncertainty Width",
  legend_title = "Uncertainty Width",
  map_data     = uq_map_sf,
  fill_colors  = uq_uncertainty_palette
)

uq_combined_map <- uq_map_predicted + uq_map_uncertainty +
  patchwork::plot_layout(ncol = 2, nrow = 1, widths = c(1, 1))

save_plot_png_tiff(
  file.path(uq_output_dir, "Predicted_Incidence_and_Uncertainty_Width_2024"),
  uq_combined_map, width = 16, height = 8, dpi = 300
)