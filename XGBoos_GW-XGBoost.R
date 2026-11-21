# GW-XGboost & G-XGboost
#Packages
library("readxl")
library("dplyr")
library("tidyr")
library("purrr")
library("tibble")
library("stringr")
library("xgboost")
library("ParBayesianOptimization")
library("geosphere")
library("writexl")

set.seed(2026)
# Data
data_path <- "C:/Users/hp/OneDrive/Desktop/D/Research_Data/Original_data_code_format.xlsx"
output    <- "C:/Users/hp/OneDrive/Desktop/D/Analysis/Recheck/GW-XGBoost_XGBoost"

N_THREADS <- max(1, parallel::detectCores() - 1)
K_CANDIDATES <- seq(10, 45, by = 5)

# Bayesian-optimisation 
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

compute_fold_metrics <- function(oof_pred) {
  map_dfr(seq_along(divisions), function(i) {
    dv <- divisions[i]
    va_idx   <- which(train_df$Division == dv)
    tr_idx   <- which(train_df$Division != dv)
    baseline <- mean(train_df$y[tr_idx])
    m <- compute_all_metrics(train_df$y[va_idx], oof_pred[va_idx], baseline)
    bind_cols(tibble(Division = dv), m)
  })
}

summarise_fold_metrics <- function(fold_metrics, model_name) {
  fold_metrics %>%
    summarise(across(c(R2, RMSE, MAE),
                     list(mean = ~mean(.x), sd = ~sd(.x)),
                     .names = "{.col}_{.fn}")) %>%
    mutate(Model = model_name, .before = 1)
}

fit_xgb <- function(data, params_list, nrounds, weight = NULL) {
  dtrain <- xgb.DMatrix(
    data   = as.matrix(data[, predictor_vars]),
    label  = data$y,
    weight = weight
  )
  xgb.train(params = params_list, data = dtrain, nrounds = nrounds, verbose = 0)
}

predict_xgb <- function(model, newdata) {
  predict(model, as.matrix(newdata[, predictor_vars]))
}

# SPATIAL CV RUNNER - GLOBAL RANDOM FOREST
# 8-fold leave-one-Division-out CV, performed on TRAIN (2017-2023) only.
run_spatial_cv_global <- function(train_data, divisions, params_list, nrounds) {
  oof_pred  <- rep(NA_real_, nrow(train_data))
  fold_rmse <- numeric(length(divisions))
  
  for (i in seq_along(divisions)) {
    dv <- divisions[i]
    tr_idx <- which(train_data$Division != dv)
    va_idx <- which(train_data$Division == dv)
    
    model <- fit_xgb(train_data[tr_idx, ], params_list, nrounds)
    preds <- predict_xgb(model, train_data[va_idx, ])
    
    oof_pred[va_idx] <- preds
    fold_rmse[i] <- calc_rmse(train_data$y[va_idx], preds)
  }
  
  list(oof_pred = oof_pred, fold_rmse = fold_rmse)
}

# GLOBAL XGBOOST - BAYESIAN HYPERPARAMETER TUNING
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
  n_estimatorsL     = c(50L, 600L),
  max_depthL        = c(2L, 10L),
  learning_rate     = c(0.01, 0.30),
  subsample         = c(0.50, 1.00),
  colsample_bytree  = c(0.50, 1.00),
  gamma             = c(0.0, 5.0),
  reg_alpha         = c(0.0, 5.0),
  reg_lambda        = c(0.1, 10.0),
  min_child_weightL = c(1L, 20L)
)

xgb_cv_score <- function(n_estimatorsL, max_depthL, learning_rate, subsample,
                         colsample_bytree, gamma, reg_alpha, reg_lambda,
                         min_child_weightL) {
  
  params_list <- list(
    objective         = "reg:squarederror",
    eval_metric       = "rmse",
    max_depth         = round(max_depthL),
    eta               = learning_rate,
    subsample         = subsample,
    colsample_bytree  = colsample_bytree,
    gamma             = gamma,
    alpha             = reg_alpha,
    lambda            = reg_lambda,
    min_child_weight  = round(min_child_weightL),
    nthread           = N_THREADS
  )
  
  cv_result <- run_spatial_cv_global(train_df, divisions, params_list, round(n_estimatorsL))
  mean_rmse <- mean(cv_result$fold_rmse)
  
  list(Score = -mean_rmse, mean_cv_rmse = mean_rmse)
}

bayes_opt_result <- run_bayes_opt_robust(
  FUN         = xgb_cv_score,
  bounds      = bayes_bounds,
  initPoints  = BAYES_INIT_POINTS,
  iters.n     = BAYES_ITERS_N,
  iters.k     = 1,
  acq         = "ei",
  base_seed   = 2026,
  verbose     = 1
)

best_pars <- getBestPars(bayes_opt_result)

best_n_estimators     <- round(best_pars$n_estimatorsL)
best_max_depth        <- round(best_pars$max_depthL)
best_learning_rate    <- best_pars$learning_rate
best_subsample        <- best_pars$subsample
best_colsample_bytree <- best_pars$colsample_bytree
best_gamma            <- best_pars$gamma
best_reg_alpha        <- best_pars$reg_alpha
best_reg_lambda       <- best_pars$reg_lambda
best_min_child_weight <- round(best_pars$min_child_weightL)

global_params_list <- list(
  objective         = "reg:squarederror",
  eval_metric       = "rmse",
  max_depth         = best_max_depth,
  eta               = best_learning_rate,
  subsample         = best_subsample,
  colsample_bytree  = best_colsample_bytree,
  gamma             = best_gamma,
  alpha             = best_reg_alpha,
  lambda            = best_reg_lambda,
  min_child_weight  = best_min_child_weight,
  nthread           = N_THREADS
)
print(global_params_list)

# GLOBAL XGBOOST
global_cv <- run_spatial_cv_global(train_df, divisions, global_params_list, best_n_estimators)
global_fold_metrics <- compute_fold_metrics(global_cv$oof_pred)
global_spatial_cv_summary <- summarise_fold_metrics(global_fold_metrics, "Global XGBoost")
global_oof_metrics <- compute_all_metrics(train_df$y, global_cv$oof_pred, mean(train_df$y)) %>%
  mutate(Model = "Global XGBoost", .before = 1)

global_final_model <- fit_xgb(train_df, global_params_list, best_n_estimators)
global_test_pred   <- predict_xgb(global_final_model, test_df)
global_test_metrics <- compute_all_metrics(test_df$y, global_test_pred, mean(train_df$y)) %>%
  mutate(Model = "Global XGBoost", .before = 1)

global_train_pred <- predict_xgb(global_final_model, train_df)
global_train_metrics <- compute_all_metrics(train_df$y, global_train_pred, mean(train_df$y)) %>%
  mutate(Model = "Global XGBoost", .before = 1)

# GW-XGBOOST - GEOGRAPHIC WEIGHTING MACHINERY
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
  dtrain <- xgb.DMatrix(
    data   = as.matrix(pool_data[use, predictor_vars]),
    label  = pool_data$y[use],
    weight = row_w[use]
  )
  xgb.train(params = global_params_list, data = dtrain,
            nrounds = best_n_estimators, verbose = 0)
}

fit_predict_local <- function(target_district, pool_data, target_rows, k, dist_mat) {
  model <- fit_local_model(target_district, pool_data, k, dist_mat)
  predict(model, as.matrix(target_rows[, predictor_vars]))
}

# GW-XGBOOST - SPATIAL CV RUNNER
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

# GW-XGBOOST - BANDWIDTH (k) SELECTION
k_grid_results <- map_dfr(K_CANDIDATES, function(k) {
  cv_res <- run_spatial_cv_gw(train_df, divisions, k, dist_matrix_km)
  tibble(k = k, mean_cv_rmse = mean(cv_res$fold_rmse), sd_cv_rmse = sd(cv_res$fold_rmse))
})
print(k_grid_results)

best_k <- k_grid_results$k[which.min(k_grid_results$mean_cv_rmse)]

# GW-XGBOOST
gw_cv <- run_spatial_cv_gw(train_df, divisions, best_k, dist_matrix_km)
gw_fold_metrics <- compute_fold_metrics(gw_cv$oof_pred)
gw_spatial_cv_summary <- summarise_fold_metrics(gw_fold_metrics, "GW-XGBoost")
gw_oof_metrics <- compute_all_metrics(train_df$y, gw_cv$oof_pred, mean(train_df$y)) %>%
  mutate(Model = "GW-XGBoost", .before = 1)

all_districts <- unique(train_df$District)
gw_test_pred  <- rep(NA_real_, nrow(test_df))
gw_train_pred <- rep(NA_real_, nrow(train_df))

for (target in all_districts) {
  local_model <- fit_local_model(target, train_df, best_k, dist_matrix_km)
  
  train_idx <- which(train_df$District == target)
  gw_train_pred[train_idx] <- predict(local_model, as.matrix(train_df[train_idx, predictor_vars]))
  
  test_idx <- which(test_df$District == target)
  if (length(test_idx) > 0) {
    gw_test_pred[test_idx] <- predict(local_model, as.matrix(test_df[test_idx, predictor_vars]))
  }
}

stopifnot(!anyNA(gw_test_pred))
stopifnot(!anyNA(gw_train_pred))

gw_test_metrics <- compute_all_metrics(test_df$y, gw_test_pred, mean(train_df$y)) %>%
  mutate(Model = "GW-XGBoost", .before = 1)

gw_train_metrics <- compute_all_metrics(train_df$y, gw_train_pred, mean(train_df$y)) %>%
  mutate(Model = "GW-XGBoost", .before = 1)

# FINAL COMPARISON
comparison <- tibble(
  Model         = c("Global XGBoost", "GW-XGBoost"),
  CV_RMSE       = c(mean(global_cv$fold_rmse), mean(gw_cv$fold_rmse)),
  OOF_RMSE      = c(global_oof_metrics$RMSE, gw_oof_metrics$RMSE),
  Test2024_RMSE = c(global_test_metrics$RMSE, gw_test_metrics$RMSE),
  Test2024_MAE  = c(global_test_metrics$MAE, gw_test_metrics$MAE),
  Test2024_R2   = c(global_test_metrics$R2, gw_test_metrics$R2)
)
print(comparison)

global_params_table <- tibble(
  Parameter = c("n_estimators", "max_depth", "learning_rate", "subsample",
                "colsample_bytree", "gamma", "reg_alpha", "reg_lambda",
                "min_child_weight"),
  Value = c(best_n_estimators, best_max_depth, best_learning_rate, best_subsample,
            best_colsample_bytree, best_gamma, best_reg_alpha, best_reg_lambda,
            best_min_child_weight)
)

gw_params_table <- bind_rows(
  global_params_table,
  tibble(Parameter = "bandwidth_k_nearest_neighbours", Value = best_k)
) %>%
  mutate(Note = ifelse(Parameter == "bandwidth_k_nearest_neighbours",
                       "Selected via spatial-CV grid search",
                       "Inherited from tuned Global XGBoost"))

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
  path = file.path(output, "Dengue_XGBoost_vs_GWXGBoost_Accuracy_Results.xlsx")
)

# Observed, Predicted, Residual

All_train_predictions <- train_df %>%
  transmute(
    Year = year,
    District = District,
    Observed = y,
    `Global XGBoost_Pred` = global_train_pred,
    `GW-XGBoost_Pred` = gw_train_pred,
    `Global XGBoost_Resi` = Observed - `Global XGBoost_Pred`,
    `GW-XGBoost_Resi` = Observed - `GW-XGBoost_Pred`
  )

All_test_predictions <- test_df %>%
  transmute(
    Year = year,
    District = District,
    Observed = y,
    `Global XGBoost_Pred` = global_test_pred,
    `GW-XGBoost_Pred` = gw_test_pred,
    `Global XGBoost_Resi` = Observed - `Global XGBoost_Pred`,
    `GW-XGBoost_Resi` = Observed - `GW-XGBoost_Pred`
  )

write_xlsx(
  list(
    All_train_predictions = All_train_predictions,
    All_test_predictions  = All_test_predictions
  ),
  path = file.path(output, "XGBoost_obs_pre_residual.xlsx")
)