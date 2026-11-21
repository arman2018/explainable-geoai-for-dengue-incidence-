# GW-LightGBM & G-LightGBM
#Packages
library("readxl")
library("dplyr")
library("tidyr")
library("purrr")
library("tibble")
library("stringr")
library("lightgbm")
library("ParBayesianOptimization")
library("geosphere")
library("writexl")

set.seed(2026)
# Data
data_path <- "C:/Users/hp/OneDrive/Desktop/D/Research_Data/Original_data_code_format.xlsx"
output <- "C:/Users/hp/OneDrive/Desktop/D/Analysis/Recheck/GW-LightGBM_LightGBM"

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

# TRAIN and TEST SPLIT
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

add_model <- function(x, model_name) mutate(x, Model = model_name, .before = 1)

get_fold_metrics <- function(oof_pred) {
  map_dfr(seq_along(divisions), function(i) {
    dv <- divisions[i]
    va_idx   <- which(train_df$Division == dv)
    tr_idx   <- which(train_df$Division != dv)
    baseline <- mean(train_df$y[tr_idx])
    m <- compute_all_metrics(train_df$y[va_idx], oof_pred[va_idx], baseline)
    bind_cols(tibble(Division = dv), m)
  })
}

get_cv_summary <- function(fold_metrics, model_name) {
  fold_metrics %>%
    summarise(across(c(R2, RMSE, MAE),
                     list(mean = ~mean(.x), sd = ~sd(.x)),
                     .names = "{.col}_{.fn}")) %>%
    add_model(model_name)
}

fit_lgb <- function(data, params_list, nrounds, weight = NULL) {
  dtrain <- lgb.Dataset(
    data   = as.matrix(data[, predictor_vars]),
    label  = data$y,
    weight = weight
  )
  lgb.train(params = params_list, data = dtrain, nrounds = nrounds, verbose = -1L)
}

predict_lgb <- function(model, newdata) {
  predict(model, as.matrix(newdata[, predictor_vars]))
}

make_params <- function(num_leaves, max_depth, learning_rate, feature_fraction,
                        bagging_fraction, min_data_in_leaf, lambda_l1, lambda_l2,
                        min_gain_to_split) {
  list(
    objective          = "regression",
    metric             = "rmse",
    num_leaves         = round(num_leaves),
    max_depth          = round(max_depth),
    learning_rate      = learning_rate,
    feature_fraction   = feature_fraction,
    bagging_fraction   = bagging_fraction,
    bagging_freq       = 1L,
    min_data_in_leaf   = round(min_data_in_leaf),
    lambda_l1          = lambda_l1,
    lambda_l2          = lambda_l2,
    min_gain_to_split  = min_gain_to_split,
    num_threads        = N_THREADS,
    verbosity          = -1L
  )
}

# SPATIAL CV RUNNER - GLOBAL LIGHTGBM
# 8-fold leave-one-Division-out CV, performed on TRAIN (2017-2023) only.
run_spatial_cv_global <- function(train_data, divisions, params_list, nrounds) {
  oof_pred  <- rep(NA_real_, nrow(train_data))
  fold_rmse <- numeric(length(divisions))
  
  for (i in seq_along(divisions)) {
    dv <- divisions[i]
    tr_idx <- which(train_data$Division != dv)
    va_idx <- which(train_data$Division == dv)
    
    model <- fit_lgb(train_data[tr_idx, ], params_list, nrounds)
    preds <- predict_lgb(model, train_data[va_idx, ])
    
    oof_pred[va_idx] <- preds
    fold_rmse[i] <- calc_rmse(train_data$y[va_idx], preds)
  }
  
  list(oof_pred = oof_pred, fold_rmse = fold_rmse)
}

# GLOBAL LIGHTGBM - BAYESIAN HYPERPARAMETER TUNING
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
  num_iterationsL    = c(50L, 600L),
  num_leavesL        = c(7L, 127L),
  max_depthL         = c(3L, 10L),
  learning_rate      = c(0.01, 0.30),
  feature_fraction   = c(0.50, 1.00),
  bagging_fraction   = c(0.50, 1.00),
  min_data_in_leafL  = c(3L, 30L),
  lambda_l1          = c(0.0, 5.0),
  lambda_l2          = c(0.1, 10.0),
  min_gain_to_split  = c(0.0, 5.0)
)

lgb_cv_score <- function(num_iterationsL, num_leavesL, max_depthL, learning_rate,
                         feature_fraction, bagging_fraction, min_data_in_leafL,
                         lambda_l1, lambda_l2, min_gain_to_split) {
  
  params_list <- make_params(num_leavesL, max_depthL, learning_rate, feature_fraction,
                             bagging_fraction, min_data_in_leafL, lambda_l1, lambda_l2,
                             min_gain_to_split)
  
  cv_result <- run_spatial_cv_global(train_df, divisions, params_list, round(num_iterationsL))
  mean_rmse <- mean(cv_result$fold_rmse)
  
  list(Score = -mean_rmse, mean_cv_rmse = mean_rmse)
}

bayes_opt_result <- run_bayes_opt_robust(
  FUN         = lgb_cv_score,
  bounds      = bayes_bounds,
  initPoints  = BAYES_INIT_POINTS,
  iters.n     = BAYES_ITERS_N,
  iters.k     = 1,
  acq         = "ei",
  base_seed   = 2026,
  verbose     = 1
)

best_pars <- getBestPars(bayes_opt_result)
best_n_estimators <- round(best_pars$num_iterationsL)

global_params_list <- make_params(
  best_pars$num_leavesL, best_pars$max_depthL, best_pars$learning_rate,
  best_pars$feature_fraction, best_pars$bagging_fraction, best_pars$min_data_in_leafL,
  best_pars$lambda_l1, best_pars$lambda_l2, best_pars$min_gain_to_split
)
print(global_params_list)

# GLOBAL LIGHTGBM
global_cv <- run_spatial_cv_global(train_df, divisions, global_params_list, best_n_estimators)
global_fold_metrics       <- get_fold_metrics(global_cv$oof_pred)
global_spatial_cv_summary <- get_cv_summary(global_fold_metrics, "Global LightGBM")

global_oof_metrics <- compute_all_metrics(train_df$y, global_cv$oof_pred, mean(train_df$y)) %>%
  add_model("Global LightGBM")
global_final_model <- fit_lgb(train_df, global_params_list, best_n_estimators)

global_test_pred <- predict_lgb(global_final_model, test_df)
global_test_metrics <- compute_all_metrics(test_df$y, global_test_pred, mean(train_df$y)) %>%
  add_model("Global LightGBM")

global_train_pred <- predict_lgb(global_final_model, train_df)
global_train_metrics <- compute_all_metrics(train_df$y, global_train_pred, mean(train_df$y)) %>%
  add_model("Global LightGBM")

# GW-LIGHTGBM - GEOGRAPHIC WEIGHTING MACHINERY
centroids <- train_df %>% distinct(District, Latitude, Longitude) %>% as.data.frame()
rownames(centroids) <- centroids$District

# Haversine great-circle distance matrix (km)
coords_lonlat <- as.matrix(centroids[, c("Longitude", "Latitude")])
dist_matrix_km <- geosphere::distm(coords_lonlat, coords_lonlat, fun = geosphere::distHaversine) / 1000
rownames(dist_matrix_km) <- centroids$District
colnames(dist_matrix_km) <- centroids$District

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
  fit_lgb(pool_data[use, ], global_params_list, best_n_estimators, weight = row_w[use])
}

# GW-LIGHTGBM - SPATIAL CV RUNNER
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
      model <- fit_local_model(target, pool_data, k, dist_mat)
      oof_pred[target_idx] <- predict_lgb(model, target_rows)
    }
    
    va_idx <- which(train_data$Division == dv)
    fold_rmse[i] <- calc_rmse(train_data$y[va_idx], oof_pred[va_idx])
  }
  
  list(oof_pred = oof_pred, fold_rmse = fold_rmse)
}

# GW-LIGHTGBM - BANDWIDTH (k) SELECTION
k_grid_results <- map_dfr(K_CANDIDATES, function(k) {
  cv_res <- run_spatial_cv_gw(train_df, divisions, k, dist_matrix_km)
  tibble(k = k, mean_cv_rmse = mean(cv_res$fold_rmse), sd_cv_rmse = sd(cv_res$fold_rmse))
})

print(k_grid_results)

best_k <- k_grid_results$k[which.min(k_grid_results$mean_cv_rmse)]

# GW-LIGHTGBM
gw_cv <- run_spatial_cv_gw(train_df, divisions, best_k, dist_matrix_km)
gw_fold_metrics       <- get_fold_metrics(gw_cv$oof_pred)
gw_spatial_cv_summary <- get_cv_summary(gw_fold_metrics, "GW-LightGBM")

gw_oof_metrics <- compute_all_metrics(train_df$y, gw_cv$oof_pred, mean(train_df$y)) %>%
  add_model("GW-LightGBM")

all_districts <- unique(train_df$District)
gw_test_pred  <- rep(NA_real_, nrow(test_df))
gw_train_pred <- rep(NA_real_, nrow(train_df))

for (target in all_districts) {
  local_model <- fit_local_model(target, train_df, best_k, dist_matrix_km)
  
  train_idx <- which(train_df$District == target)
  gw_train_pred[train_idx] <- predict_lgb(local_model, train_df[train_idx, ])
  
  test_idx <- which(test_df$District == target)
  if (length(test_idx) > 0) {
    gw_test_pred[test_idx] <- predict_lgb(local_model, test_df[test_idx, ])
  }
}

stopifnot(!anyNA(gw_test_pred))
stopifnot(!anyNA(gw_train_pred))

gw_test_metrics <- compute_all_metrics(test_df$y, gw_test_pred, mean(train_df$y)) %>%
  add_model("GW-LightGBM")

gw_train_metrics <- compute_all_metrics(train_df$y, gw_train_pred, mean(train_df$y)) %>%
  add_model("GW-LightGBM")

# FINAL COMPARISON
comparison <- tibble(
  Model         = c("Global LightGBM", "GW-LightGBM"),
  CV_RMSE       = c(mean(global_cv$fold_rmse), mean(gw_cv$fold_rmse)),
  OOF_RMSE      = c(global_oof_metrics$RMSE, gw_oof_metrics$RMSE),
  Test2024_RMSE = c(global_test_metrics$RMSE, gw_test_metrics$RMSE),
  Test2024_MAE  = c(global_test_metrics$MAE, gw_test_metrics$MAE),
  Test2024_R2   = c(global_test_metrics$R2, gw_test_metrics$R2)
)

print(comparison)

global_params_table <- tibble(
  Parameter = c("num_iterations_nrounds", "num_leaves", "max_depth", "learning_rate",
                "feature_fraction", "bagging_fraction", "bagging_freq",
                "min_data_in_leaf", "lambda_l1", "lambda_l2", "min_gain_to_split"),
  Value = c(best_n_estimators, global_params_list$num_leaves, global_params_list$max_depth,
            global_params_list$learning_rate, global_params_list$feature_fraction,
            global_params_list$bagging_fraction, global_params_list$bagging_freq,
            global_params_list$min_data_in_leaf, global_params_list$lambda_l1,
            global_params_list$lambda_l2, global_params_list$min_gain_to_split)
)

gw_params_table <- bind_rows(
  global_params_table,
  tibble(Parameter = "bandwidth_k_nearest_neighbours", Value = best_k)
) %>%
  mutate(Note = ifelse(Parameter == "bandwidth_k_nearest_neighbours",
                       "Selected via spatial-CV grid search",
                       "Inherited from tuned Global LightGBM"))

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
  path = file.path(output, "Dengue_LightGBM_vs_GWLightGBM_Accuracy_Results.xlsx")
)

# Observed, Predicted, Residual

make_pred_table <- function(data, global_pred, gw_pred) {
  data %>%
    transmute(
      Year = year,
      District = District,
      Observed = y,
      `Global LightGBM_Pred` = global_pred,
      `GW-LightGBM_Pred` = gw_pred,
      `Global LightGBM_Resi` = Observed - `Global LightGBM_Pred`,
      `GW-LightGBM_Resi` = Observed - `GW-LightGBM_Pred`
    )
}

All_train_predictions <- make_pred_table(train_df, global_train_pred, gw_train_pred)
All_test_predictions  <- make_pred_table(test_df, global_test_pred, gw_test_pred)

write_xlsx(
  list(
    All_train_predictions = All_train_predictions,
    All_test_predictions  = All_test_predictions
  ),
  path = file.path(output, "LightGBM_obs_pre_residual.xlsx")
)