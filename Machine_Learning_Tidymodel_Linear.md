Machine_Learning_Tidymodel_Linear
================
Bernard Asante
2025-02-16

``` r
data <- read_csv("Heart Attack.csv")
```

    ## Rows: 1319 Columns: 9
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (1): class
    ## dbl (8): age, gender, impluse, pressurehight, pressurelow, glucose, kcm, tro...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
data %>% 
  glimpse()
```

    ## Rows: 1,319
    ## Columns: 9
    ## $ age           <dbl> 64, 21, 55, 64, 55, 58, 32, 63, 44, 67, 44, 63, 64, 54, …
    ## $ gender        <dbl> 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0,…
    ## $ impluse       <dbl> 66, 94, 64, 70, 64, 61, 40, 60, 60, 61, 60, 60, 60, 94, …
    ## $ pressurehight <dbl> 160, 98, 160, 120, 112, 112, 179, 214, 154, 160, 166, 15…
    ## $ pressurelow   <dbl> 83, 46, 77, 55, 65, 58, 68, 82, 81, 95, 90, 83, 99, 67, …
    ## $ glucose       <dbl> 160, 296, 270, 270, 300, 87, 102, 87, 135, 100, 102, 198…
    ## $ kcm           <dbl> 1.800, 6.750, 1.990, 13.870, 1.080, 1.830, 0.710, 300.00…
    ## $ troponin      <dbl> 0.012, 1.060, 0.003, 0.122, 0.003, 0.004, 0.003, 2.370, …
    ## $ class         <chr> "negative", "positive", "negative", "positive", "negativ…

``` r
data <- data %>% 
  rename(pulse = impluse, sys = pressurehight, dias = pressurelow) %>% 
  mutate(glu_mmol = glucose/18)

data %>% 
  glimpse()
```

    ## Rows: 1,319
    ## Columns: 10
    ## $ age      <dbl> 64, 21, 55, 64, 55, 58, 32, 63, 44, 67, 44, 63, 64, 54, 47, 6…
    ## $ gender   <dbl> 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1…
    ## $ pulse    <dbl> 66, 94, 64, 70, 64, 61, 40, 60, 60, 61, 60, 60, 60, 94, 76, 8…
    ## $ sys      <dbl> 160, 98, 160, 120, 112, 112, 179, 214, 154, 160, 166, 150, 19…
    ## $ dias     <dbl> 83, 46, 77, 55, 65, 58, 68, 82, 81, 95, 90, 83, 99, 67, 70, 6…
    ## $ glucose  <dbl> 160, 296, 270, 270, 300, 87, 102, 87, 135, 100, 102, 198, 92,…
    ## $ kcm      <dbl> 1.800, 6.750, 1.990, 13.870, 1.080, 1.830, 0.710, 300.000, 2.…
    ## $ troponin <dbl> 0.012, 1.060, 0.003, 0.122, 0.003, 0.004, 0.003, 2.370, 0.004…
    ## $ class    <chr> "negative", "positive", "negative", "positive", "negative", "…
    ## $ glu_mmol <dbl> 8.888889, 16.444444, 15.000000, 15.000000, 16.666667, 4.83333…

## Recoding variables

``` r
data <- data %>% 
  mutate(gender = case_when(
    gender == 1 ~ "Male",
    gender == 0 ~ "Female"
  )) %>% 
  mutate(gender = as.factor(gender)) %>% 
  mutate(class = as.factor(class))

data %>% 
  glimpse()
```

    ## Rows: 1,319
    ## Columns: 10
    ## $ age      <dbl> 64, 21, 55, 64, 55, 58, 32, 63, 44, 67, 44, 63, 64, 54, 47, 6…
    ## $ gender   <fct> Male, Male, Male, Male, Male, Female, Female, Male, Female, M…
    ## $ pulse    <dbl> 66, 94, 64, 70, 64, 61, 40, 60, 60, 61, 60, 60, 60, 94, 76, 8…
    ## $ sys      <dbl> 160, 98, 160, 120, 112, 112, 179, 214, 154, 160, 166, 150, 19…
    ## $ dias     <dbl> 83, 46, 77, 55, 65, 58, 68, 82, 81, 95, 90, 83, 99, 67, 70, 6…
    ## $ glucose  <dbl> 160, 296, 270, 270, 300, 87, 102, 87, 135, 100, 102, 198, 92,…
    ## $ kcm      <dbl> 1.800, 6.750, 1.990, 13.870, 1.080, 1.830, 0.710, 300.000, 2.…
    ## $ troponin <dbl> 0.012, 1.060, 0.003, 0.122, 0.003, 0.004, 0.003, 2.370, 0.004…
    ## $ class    <fct> negative, positive, negative, positive, negative, negative, n…
    ## $ glu_mmol <dbl> 8.888889, 16.444444, 15.000000, 15.000000, 16.666667, 4.83333…

## Exploring missing data

``` r
missing_data <- data %>% 
  summarise_all(~sum(is.na(.)))

missing_data
```

    ## # A tibble: 1 × 10
    ##     age gender pulse   sys  dias glucose   kcm troponin class glu_mmol
    ##   <int>  <int> <int> <int> <int>   <int> <int>    <int> <int>    <int>
    ## 1     0      0     0     0     0       0     0        0     0        0

# **Predictor Analysis**

## EDA of the predictor

``` r
ggplot(data, aes (troponin ))+
  geom_histogram()
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](Machine_Learning_Tidymodel_Linear_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
univ_table <- data %>% 
  tbl_uvregression(
    method = lm,
    y = troponin
  )

univ_table %>% as_kable()
```

| **Characteristic** | **N** | **Beta** | **95% CI**  | **p-value** |
|:-------------------|:-----:|:--------:|:-----------:|:-----------:|
| age                | 1,319 |   0.01   | 0.00, 0.01  |    0.001    |
| gender             | 1,319 |          |             |             |
| Female             |       |    —     |      —      |             |
| Male               |       |   0.16   | 0.03, 0.29  |    0.017    |
| pulse              | 1,319 |   0.00   | 0.00, 0.00  |     0.7     |
| sys                | 1,319 |   0.00   | 0.00, 0.00  |    0.11     |
| dias               | 1,319 |   0.00   | 0.00, 0.01  |    0.12     |
| glucose            | 1,319 |   0.00   | 0.00, 0.00  |     0.4     |
| kcm                | 1,319 |   0.00   | 0.00, 0.00  |     0.6     |
| class              | 1,319 |          |             |             |
| negative           |       |    —     |      —      |             |
| positive           |       |   0.54   | 0.42, 0.67  |   \<0.001   |
| glu_mmol           | 1,319 |   0.01   | -0.01, 0.02 |     0.4     |

## Correlation

### Correlation_matrix

``` r
library(reshape2)
```

    ## 
    ## Attaching package: 'reshape2'

    ## The following object is masked from 'package:tidyr':
    ## 
    ##     smiths

``` r
corr_matrix <- data %>% 
  select(age, pulse, glu_mmol,kcm,troponin) %>% 
  cor()

corr_matrix
```

    ##                   age       pulse     glu_mmol         kcm    troponin
    ## age       1.000000000 -0.02343985 -0.004192985  0.01841874  0.08880023
    ## pulse    -0.023439847  1.00000000 -0.019583957 -0.01300104  0.01117985
    ## glu_mmol -0.004192985 -0.01958396  1.000000000  0.04575658  0.02106887
    ## kcm       0.018418737 -0.01300104  0.045756581  1.00000000 -0.01600835
    ## troponin  0.088800234  0.01117985  0.021068866 -0.01600835  1.00000000

``` r
melted_matrix <- melt(corr_matrix)

melted_matrix
```

    ##        Var1     Var2        value
    ## 1       age      age  1.000000000
    ## 2     pulse      age -0.023439847
    ## 3  glu_mmol      age -0.004192985
    ## 4       kcm      age  0.018418737
    ## 5  troponin      age  0.088800234
    ## 6       age    pulse -0.023439847
    ## 7     pulse    pulse  1.000000000
    ## 8  glu_mmol    pulse -0.019583957
    ## 9       kcm    pulse -0.013001040
    ## 10 troponin    pulse  0.011179854
    ## 11      age glu_mmol -0.004192985
    ## 12    pulse glu_mmol -0.019583957
    ## 13 glu_mmol glu_mmol  1.000000000
    ## 14      kcm glu_mmol  0.045756581
    ## 15 troponin glu_mmol  0.021068866
    ## 16      age      kcm  0.018418737
    ## 17    pulse      kcm -0.013001040
    ## 18 glu_mmol      kcm  0.045756581
    ## 19      kcm      kcm  1.000000000
    ## 20 troponin      kcm -0.016008351
    ## 21      age troponin  0.088800234
    ## 22    pulse troponin  0.011179854
    ## 23 glu_mmol troponin  0.021068866
    ## 24      kcm troponin -0.016008351
    ## 25 troponin troponin  1.000000000

``` r
ggplot(melted_matrix, aes(x = Var1, y = Var2, fill = value))+
  geom_tile()+
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0)
```

![](Machine_Learning_Tidymodel_Linear_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
pairs.panels( data,
              scale = FALSE,
              method = "spearman",
              lm = TRUE,
              cor = TRUE,
              jiggle = FALSE,
              stars = FALSE)
```

![](Machine_Learning_Tidymodel_Linear_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

# **Machine Leaning**

## Selecting variable

``` r
lin_data <- data %>% 
  select(age,gender, pulse,sys, dias,kcm,glu_mmol,troponin, class)

lin_data %>% 
  glimpse()
```

    ## Rows: 1,319
    ## Columns: 9
    ## $ age      <dbl> 64, 21, 55, 64, 55, 58, 32, 63, 44, 67, 44, 63, 64, 54, 47, 6…
    ## $ gender   <fct> Male, Male, Male, Male, Male, Female, Female, Male, Female, M…
    ## $ pulse    <dbl> 66, 94, 64, 70, 64, 61, 40, 60, 60, 61, 60, 60, 60, 94, 76, 8…
    ## $ sys      <dbl> 160, 98, 160, 120, 112, 112, 179, 214, 154, 160, 166, 150, 19…
    ## $ dias     <dbl> 83, 46, 77, 55, 65, 58, 68, 82, 81, 95, 90, 83, 99, 67, 70, 6…
    ## $ kcm      <dbl> 1.800, 6.750, 1.990, 13.870, 1.080, 1.830, 0.710, 300.000, 2.…
    ## $ glu_mmol <dbl> 8.888889, 16.444444, 15.000000, 15.000000, 16.666667, 4.83333…
    ## $ troponin <dbl> 0.012, 1.060, 0.003, 0.122, 0.003, 0.004, 0.003, 2.370, 0.004…
    ## $ class    <fct> negative, positive, negative, positive, negative, negative, n…

``` r
lin_data %>% 
  summary()
```

    ##       age            gender        pulse              sys       
    ##  Min.   : 14.00   Female:449   Min.   :  20.00   Min.   : 42.0  
    ##  1st Qu.: 47.00   Male  :870   1st Qu.:  64.00   1st Qu.:110.0  
    ##  Median : 58.00                Median :  74.00   Median :124.0  
    ##  Mean   : 56.19                Mean   :  78.34   Mean   :127.2  
    ##  3rd Qu.: 65.00                3rd Qu.:  85.00   3rd Qu.:143.0  
    ##  Max.   :103.00                Max.   :1111.00   Max.   :223.0  
    ##       dias             kcm             glu_mmol         troponin      
    ##  Min.   : 38.00   Min.   :  0.321   Min.   : 1.944   Min.   : 0.0010  
    ##  1st Qu.: 62.00   1st Qu.:  1.655   1st Qu.: 5.444   1st Qu.: 0.0060  
    ##  Median : 72.00   Median :  2.850   Median : 6.444   Median : 0.0140  
    ##  Mean   : 72.27   Mean   : 15.274   Mean   : 8.146   Mean   : 0.3609  
    ##  3rd Qu.: 81.00   3rd Qu.:  5.805   3rd Qu.: 9.417   3rd Qu.: 0.0855  
    ##  Max.   :154.00   Max.   :300.000   Max.   :30.056   Max.   :10.3000  
    ##       class    
    ##  negative:509  
    ##  positive:810  
    ##                
    ##                
    ##                
    ## 

## Splitting Data

``` r
set.seed(123)

data_split <- initial_split(lin_data, prop = 0.70)

train_data <- training(data_split)

test_data <- testing(data_split)
```

## Building Model

``` r
lin_ridge <- linear_reg(penalty= tune(), mixture = tune()) %>% 
  set_engine("glmnet")
```

## Building Recipe

``` r
lin_recipe <- recipe(troponin ~ ., data = train_data) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors())
```

## Building workflow

``` r
lin_workflow <- workflow() %>% 
  add_model(lin_ridge) %>% 
  add_recipe(lin_recipe)
```

## Hyperparameter Tunning

``` r
fold = vfold_cv(train_data, v=5)

grid <- grid_regular(
  penalty(range = c(0.0001, 1)),  
  mixture(range = c(0, 1)),       # Mixture is between 0 (ridge) and 1 (lasso)
  levels = c(5, 5)                # Adjust the number of grid points
)


lin_tune_wrkflow <- tune_grid(lin_workflow,
                              resamples = fold,
                              grid = grid
)
```

    ## → A | warning: A correlation computation is required, but `estimate` is constant and has 0
    ##                standard deviation, resulting in a divide by 0 error. `NA` will be returned.

    ## There were issues with some computations   A: x1There were issues with some computations   A: x2There were issues with some computations   A: x3There were issues with some computations   A: x4There were issues with some computations   A: x5There were issues with some computations   A: x5

``` r
lin_tune_wrkflow
```

    ## # Tuning results
    ## # 5-fold cross-validation 
    ## # A tibble: 5 × 4
    ##   splits            id    .metrics          .notes          
    ##   <list>            <chr> <list>            <list>          
    ## 1 <split [738/185]> Fold1 <tibble [50 × 6]> <tibble [1 × 3]>
    ## 2 <split [738/185]> Fold2 <tibble [50 × 6]> <tibble [1 × 3]>
    ## 3 <split [738/185]> Fold3 <tibble [50 × 6]> <tibble [1 × 3]>
    ## 4 <split [739/184]> Fold4 <tibble [50 × 6]> <tibble [1 × 3]>
    ## 5 <split [739/184]> Fold5 <tibble [50 × 6]> <tibble [1 × 3]>
    ## 
    ## There were issues with some computations:
    ## 
    ##   - Warning(s) x5: A correlation computation is required, but `estimate` is constant...
    ## 
    ## Run `show_notes(.Last.tune.result)` for more information.

## Selecting and Finalizing the best model

``` r
best_params <- select_best(lin_tune_wrkflow, metric = "rmse")  # Change to "accuracy" for classification

final_wrkflow <- finalize_workflow(lin_workflow, best_params)
```

## Training the model

``` r
final_wrkflow_fit <- final_wrkflow %>% 
  fit(train_data)
```

``` r
final_wrkflow_fit %>% 
  extract_fit_parsnip() %>% 
  tidy()
```

    ## # A tibble: 9 × 3
    ##   term           estimate penalty
    ##   <chr>             <dbl>   <dbl>
    ## 1 (Intercept)     0.122      1.00
    ## 2 age             0.0393     1.00
    ## 3 pulse           0.00787    1.00
    ## 4 sys             0.0189     1.00
    ## 5 dias            0.0156     1.00
    ## 6 kcm            -0.0177     1.00
    ## 7 glu_mmol        0.0316     1.00
    ## 8 gender_Male     0.0752     1.00
    ## 9 class_positive  0.276      1.00

## Predictions

``` r
predict <- predict(final_wrkflow_fit, new_data = test_data)

predict %>% 
  head()
```

    ## # A tibble: 6 × 1
    ##    .pred
    ##    <dbl>
    ## 1 0.266 
    ## 2 0.525 
    ## 3 0.0669
    ## 4 0.196 
    ## 5 0.0936
    ## 6 0.242

## Model Evaluation

``` r
model_eval <- bind_cols(test_data, predict)


model_eval %>% 
  glimpse()
```

    ## Rows: 396
    ## Columns: 10
    ## $ age      <dbl> 64, 64, 32, 63, 54, 47, 60, 48, 63, 35, 68, 50, 40, 38, 63, 5…
    ## $ gender   <fct> Male, Male, Female, Female, Female, Male, Male, Male, Male, M…
    ## $ pulse    <dbl> 66, 70, 40, 60, 94, 76, 92, 135, 66, 62, 61, 96, 87, 80, 81, …
    ## $ sys      <dbl> 160, 120, 179, 150, 122, 120, 151, 98, 135, 137, 121, 105, 11…
    ## $ dias     <dbl> 83, 55, 68, 83, 67, 70, 78, 60, 55, 61, 49, 70, 78, 78, 65, 6…
    ## $ kcm      <dbl> 1.800, 13.870, 0.710, 2.390, 1.420, 2.570, 1.600, 94.790, 0.4…
    ## $ glu_mmol <dbl> 8.888889, 15.000000, 5.666667, 11.000000, 5.388889, 17.722222…
    ## $ troponin <dbl> 0.012, 0.122, 0.003, 0.013, 0.012, 0.003, 0.005, 0.004, 10.00…
    ## $ class    <fct> negative, positive, negative, negative, negative, negative, n…
    ## $ .pred    <dbl> 0.26578745, 0.52472645, 0.06688464, 0.19574405, 0.09361753, 0…

## Accuracy

``` r
rmse <- rmse(model_eval, truth = troponin, estimate = .pred)

rmse
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 rmse    standard        1.32
