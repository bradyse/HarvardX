# CAPSTONE: MOVIELENS PROJECT ------------------------------------------------------------
## HarvardX Data Science Professional Certificate
## Sara E. Brady, Ph.D.
## https://www.linkedin.com/in/sara-brady/

# LIBRARIES --------------------------------------------------------------------
library(tidyverse)
library(caret)
library(patchwork)
library(data.table)
library(Matrix)
library(irlba)
library(doParallel)

# CREATE EDX AND FINAL HOLD OUT DATASETS ---------------------------------------
# The code below was developed by the HarvardX Team for the purpose of creating
# the main testing data set (edx) and the final testing dataset 
# (final_holdout_test).

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Save files in project data folder 
saveRDS(edx, file = "9_Capstone/data/edx.rds")
saveRDS(final_holdout_test, file = "9_Capstone/data/final_holdout_test.rds") 

# EXPLORATORY DATA ANALYSIS ----------------------------------------------------
# Explore data structure and main variables in edx dataset.

## Basic Structure -------------------------------------------------------------

# Examine structure of edX dataset
str(edx)

## Distinct Values -------------------------------------------------------------

# Number of distinct users, movies, and genres
edx_ns <- edx |> 
  reframe(n_users = n_distinct(userId),
          n_movies = n_distinct(movieId),
          n_genres = n_distinct(genres))

edx_ns

## Ratings ---------------------------------------------------------------------

# Range of possible ratings
range(edx$rating)

# Histogram of all user ratings
mu <- mean(edx$rating)

edx |> 
  group_by(rating) |> 
  count() |> 
  ggplot() +
  geom_col(aes(rating, n)) +
  geom_vline(xintercept = mu,
             linetype = 2) +
  annotate(geom = "text", x = 2.8, y = 3000000,
           label = paste0("Mean = ", round(mu, 3)),
           color = "red")

## Ratings per User ------------------------------------------------------------

# Calculate descriptive statistics on the number of ratings per user
ratings_users <- edx |> 
  count(userId) |> 
  reframe(N = n(),
          Mean = mean(n),
          Median = median(n),
          SD = sd(n),
          Min = min(n),
          Max = max(n),
          IQR.25 = quantile(n, 0.25),
          IQR.75 = quantile(n, 0.75)) |> 
  pivot_longer(everything(), names_to = "statistic")

# Observe the large positive skew (comparing the mean to median)
ratings_users

# Histogram of the number of ratings per user
avg_ratings_per_user <- ratings_users$value[which(ratings_users$statistic=="Mean")]

count(edx, userId) |> 
  ggplot() +
  geom_histogram(aes(n), 
                 bins = 60,
                 color = "lightgray") +
  scale_x_log10() + # log-transform x-axis because variable is heavily skewed
  geom_vline(xintercept = avg_ratings_per_user,
             linetype = 2) +
  annotate(geom = "text", x = 350, y = 3500,
           label = paste0("Mean = ", round(avg_ratings_per_user, 3)),
           color = "red")

## Ratings per Movie -----------------------------------------------------------

# Calculate descriptive statistics on the number of ratings per movie
ratings_movies <- edx |> 
  count(movieId)

ratings_movies_summary <- ratings_movies |> 
  reframe(N = n(),
          Mean = mean(n),
          Median = median(n),
          SD = sd(n),
          Min = min(n),
          Max = max(n),
          IQR.25 = quantile(n, 0.25),
          IQR.75 = quantile(n, 0.75)) |> 
  pivot_longer(everything(), names_to = "statistic")

# Notice the positive skew (comparing mean to median)
ratings_movies_summary

# How many movies had only one rating?
filter(ratings_movies, n == 1) |> 
  nrow()

# How many ratings did the most frequently-rated movie have?
most_ratings <- ratings_movies |> 
  left_join(select(edx, movieId, title) |> distinct(), 
            by = "movieId") |> 
  filter(n == max(n))

most_ratings

# Histogram of the average number of ratings per movie
avg_ratings_per_movie <- ratings_movies_summary |> 
  filter(statistic == "Mean") |> 
  pull(value)

count(edx, movieId) |> 
  ggplot() +
  geom_histogram(aes(n), 
                 bins = 60,
                 color = "lightgray") +
  scale_x_log10() +
  geom_vline(xintercept = avg_ratings_per_movie,
             linetype = 2) +
  annotate(geom = "text", x = 4275, y = 390,
           label = paste0("Mean = ", round(avg_ratings_per_movie, 3)),
           color = "red")


# CREATING MAIN TEST AND TRAINING SETS -----------------------------------------
# Because we can only use the edx dataset, we need to create subsets of test and
# training datasets in the same way we created the edx and final_holdout_test 
# datasets.

y = edx$rating
set.seed(7)
test_index <- caret::createDataPartition(y, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

test_set <- temp |> 
  semi_join(train_set, by = "movieId") |> 
  semi_join(train_set, by = "userId")

removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

# Save these data files for later use
saveRDS(train_set, "9_Capstone/data/train-set.rds")
saveRDS(test_set,  "9_Capstone/data/test-set.rds")

# TRAINING ALGORITHMS ----------------------------------------------------------
# Before training algorithms, we need to write function for RMSE and benchmark
# all future models against the average (naive) model.

## RMSE ------------------------------------------------------------------------

# For testing model accuracy, store the RMSE formula into a function for later.
RMSE <- function(actual_ratings, predicted_ratings){
  sqrt(mean((actual_ratings - predicted_ratings)^2))
}

## 0. Average Model (Naive Model) ----------------------------------------------

# Mean rating of all rows in train_set
# This is our predicted value for the naive model
mu_hat <- mean(train_set$rating)

# Calculate naive RMSE
naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

# Plot residuals of the naive RMSE model
naive_df <- select(test_set, rating) |> 
  mutate(residuals = rating - mu_hat)

# Notice how this plot looks identical to the histogram of all ratings
# That is because we have now mean-centered the ratings
select(naive_df, residuals) |> 
  ggplot(aes(x = residuals)) +
  geom_histogram(binwidth = 0.5,
                 color = "lightgray") +
  geom_vline(xintercept = 0,
             linetype = 2)

## 1. Movie Effect Model -------------------------------------------------------
# To remove movie effect (bias) from model, we can estimate least squared 
# estimates like with ordinary least squares regression. Instead of running the 
# model below, we will estimate the least squares by taking the average of each 
# user's ratings of movie i and subtracting it from the average (mu_hat).

# DO NOT RUN
# fit <- lm(rating ~ movieId, data = train_set)

# Calculate movie averages
movie_avgs <- train_set |>  
  group_by(movieId) |>  
  summarize(avg_rating = mean(rating),
            b_i_hat = mean(rating - mu_hat))

# Calculate predicted ratings
predicted_ratings <- mu_hat + test_set |> 
  left_join(movie_avgs, by="movieId") |> 
  pull(b_i_hat)

# Calculate RMSE
fit1_rmse <- RMSE(predicted_ratings, test_set$rating)
fit1_rmse

# After estimating movie averages, we can explore how well our model predicts
# user's scores based upon the movie's average. To start, we determine based 
# upon our model, what are the highest-rated movies?
top_movies <- movie_avgs |> 
  filter(b_i_hat > quantile(b_i_hat, .99)) |> # Choose top 1% of movies
  pull(movieId)

top1perc <- train_set |> 
  left_join(movie_avgs, by = "movieId") |> 
  filter(movieId %in% top_movies) |> 
  group_by(movieId, title) |> 
  reframe(n = n(), avg_rating) |> 
  distinct()

# Number of top-rated movies
nrow(top1perc)

# Range of ratings for top-rated movies
range(top1perc$avg_rating)

# Random sample of top-rated movie titles
# Notice how some of them were only rated a few times, whereas others were much
# more popular.
top1perc |> 
  slice_sample(n = 10)

# Histogram of the number of ratings per top-rated movie
# Notice how most of them were only rated once
top1perc |> 
  ggplot(aes(n)) +
  geom_histogram(binwidth = 500,
                 color = "lightgray") +
  labs(title = "Frequency of Ratings for Top 1% of Movies",
       x = "Number of Ratings per Movie")

# Plot residuals against predicted ratings. The residuals look a bit strange
# due to the discrete scale of measurement, but the best fit line is still flat.
fit1_resid <- test_set |> 
  left_join(movie_avgs, by='movieId') |> 
  mutate(predicted_ratings = mu_hat + b_i_hat,
         residuals = rating - predicted_ratings)

p_fit1_resid <- fit1_resid |> 
  ggplot(aes(x = predicted_ratings, y = residuals)) +
  geom_point() +
  geom_smooth() +
  labs(title = "Residual Plot for Movie Effects Model")

p_fit1_resid

## 2. Movie & User Effects Model -----------------------------------------------
# Now we can remove both user and movie effects too see how the model improves.

# Calculate user bias by estimating least squares of user averages
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u_hat = mean(rating - mu_hat - b_i_hat))

# Write clamp function to restrict ratings between 0.5 and 5.0
clamp <- function(x, min = 0.5, max = 5) pmax(pmin(x, max), min)

# Predicted ratings by adding both movie and user effects to global average
predicted_ratings <- test_set |> 
  left_join(movie_avgs, by="movieId") |> 
  left_join(user_avgs, by="userId") |> 
  mutate(pred = clamp(mu_hat + b_i_hat + b_u_hat)) %>%
  pull(pred)

fit2_rmse <- RMSE(predicted_ratings, test_set$rating)
fit2_rmse

# Plot residuals against predicted ratings. Outliers are removed and best fit
# line is still flat, but we still didn't account for the fact that some rare
# movies are getting ratings that are off because of the small sample size.
fit2_resid <- test_set |> 
  left_join(movie_avgs, by="movieId") |> 
  left_join(user_avgs, by="userId") |> 
  mutate(predicted_ratings = clamp(mu_hat + b_i_hat + b_u_hat),
         residuals = rating - predicted_ratings)

fit2_resid |> 
  ggplot(aes(x = predicted_ratings, y = residuals)) +
  geom_point() +
  geom_smooth() +
  labs(title = "Residual Plot for Movie & User Effects Model")

## 3. Regularized Movie & User Effect Model ------------------------------------
# To write a model that could ignore movie averages with a few user ratings and 
# replace those predicted ratings with the global mean, we need to test a model
# that regularizes the movie and user averages.

# Write function that adds a penalty term to denominator when estimating movie
# and user averages and returns the RMSE.

# Different levels of lambda that we want to test
lambdas <- seq(0, 10, 0.25)

# Track processing time (this model should take less than 2 minutes)
t1 <- Sys.time()
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  
  # Regularized movie effects
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l)) # lambda is added to denominator
  
  # Regularized user effects
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l)) # lambda added
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = clamp(mu + b_i + b_u)) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

Sys.time() - t1

# Plot RMSE against different levels of lambda
ggplot(data.frame(lambdas, rmses)) +
  geom_point(aes(lambdas, rmses))

# What was the best lambda that resulted in the lowest RMSE?
lambda_m3 <- lambdas[which.min(rmses)]
lambda_m3

fit3_rmse <- min(rmses)
fit3_rmse

## 4. SVD + Movie Effect Model -------------------------------------------------
# Because we still have too many predictors (movies) in the model, we can use
# singular value decomposition (similar to principal components analysis) to 
# reduce the numbers of dimensions in the model. SVD can identify the
# interaction patterns between users and movies.

### Rationale for SVD ----------------------------------------------------------
#### Exploring Residuals -------------------------------------------------------
# To explore whether SVD makes, sense, we need to explore residuals of specific
# movies (predictors) to determine whether relationships exist between the
# residuals of one movie and the residuals of another. If residuals show any
# type of non-zero relationship, then the model is not capturing commonalities
# between movies and users.

# Based upon the regularized model above fit3_rmse, what were some highly-rated 
# movies that had relatively good ratings?

# Since we did not save the predicted values in the function above, we calculate
# them again here by first calculating the movie bias.
fit3_b_i_hat <- train_set |> 
  group_by(movieId) |> 
  summarize(b_i_hat = sum(rating - mu_hat)/(n()+lambda_m3))

# Then, calculate the user bias.
fit3_b_u_hat <- train_set |> 
  left_join(fit3_b_i_hat, by="movieId") |> 
  group_by(userId) |> 
  summarize(b_u_hat = sum(rating - b_i_hat - mu_hat)/(n()+lambda_m3))

# Predicted ratings and residuals based upon the regularized movie + user
# effects model
fit3_resid <- test_set |> 
  left_join(movie_avgs, by="movieId") |> 
  left_join(user_avgs, by="userId") |> 
  mutate(predicted_ratings = clamp(mu_hat + b_i_hat + b_u_hat),
         residuals = rating - predicted_ratings)

# What were the most popular movie IDs?
frequently_rated_films <- fit3_resid |> 
  group_by(movieId) |> 
  reframe(n = n(), title, b_i_hat, genres) |> 
  distinct() |> 
  filter(b_i_hat > 0 & n > 1000)

popular_movieIds <- frequently_rated_films |> pull(movieId)

# Pivot model's data frame into a wide format matrix such that the movies are
# the columns and the values are the residuals
popular_wide <- fit3_resid |> 
  filter(movieId %in% popular_movieIds) |> 
  pivot_wider(id_cols = c(userId),
              names_from = movieId,
              values_from = residuals,
              names_prefix = "m")

# Use Spearman's correlation to calculate the correlation of the residuals 
# between all popular movies and filter only strong correlations.
popular_rs <- cor(popular_wide[,-1], 
                  method = "spearman",
                  use = "pairwise.complete.obs") |> 
  as.data.frame() |> 
  rownames_to_column() |> 
  pivot_longer(-rowname) |> 
  filter(between(value, .6, .99)) |> 
  arrange(-value) |> 
  rename(x = rowname, y = name) |> 
  mutate(across(c(1:2), ~ as.integer(str_remove(., "m"))))

# To see the movie titles of these correlated films, join them with the long
# format version of the data above
popular_movies_rs <- left_join(popular_rs, 
                               select(frequently_rated_films, movieId, title),
                               by = c("x" = "movieId")) |> 
  rename(x_title = title) |> 
  left_join(select(frequently_rated_films, movieId, title),
            by = c("y" = "movieId")) |> 
  rename(y_title = title)

# After inspecting the movies that were highly correlated, the below plots
# show four such pairings where the residuals are positively correlated.
p1 <- fit3_resid |> 
  filter(movieId %in% c("260", "1210")) |> 
  pivot_wider(id_cols = c(userId),
              names_from = movieId,
              values_from = residuals,
              names_prefix = "m") |> 
  na.omit() |> 
  ggplot(aes(x = m260, y = m1210)) +
  geom_point() +
  geom_smooth() +
  labs(x = "A New Hope (a.k.a. Star Wars) (1977)",
       y = "Return of the Jedi (1983)",
       title = "Same Movie Franchise",
       subtitle = "Star Wars")

p2 <- fit3_resid |> 
  filter(movieId %in% c("25", "1704")) |> 
  pivot_wider(id_cols = c(userId),
              names_from = movieId,
              values_from = residuals,
              names_prefix = "m") |> 
  na.omit() |> 
  ggplot(aes(x = m1704, y = m25)) +
  geom_point() +
  geom_smooth() +
  labs(x = "Good Will Hunting (1997)",
       y = "Leaving Las Vegas (1995)",
       title = "Same Genre",
       subtitle = "Drama|Romance")

p3 <- fit3_resid |> 
  filter(movieId %in% c("25", "2959")) |> 
  pivot_wider(id_cols = c(userId),
              names_from = movieId,
              values_from = residuals,
              names_prefix = "m") |> 
  na.omit() |> 
  ggplot(aes(x = m2959, y = m25)) +
  geom_point() +
  geom_smooth()+
  labs(x = "Fight Club (1999)",
       y = "Leaving Las Vegas (1995)",
       title = "Overlapping Genres",
       subtitle = "Action|Crime|Drama|Thriller vs. Drama|Romance")

p4 <- fit3_resid |> 
  filter(movieId %in% c("924", "1208")) |> 
  pivot_wider(id_cols = c(userId),
              names_from = movieId,
              values_from = residuals,
              names_prefix = "m") |> 
  na.omit() |> 
  ggplot(aes(x = m924, y = m1208)) +
  geom_point() +
  geom_smooth() +
  labs(x = "2001: A Space Odyssey (1968)",
       y = "Apocalypse Now (1979)",
       title = "Non-overlapping Genres",
       subtitle = "Adventure|Sci-Fi vs. Action|Drama|War")

# Create one data visualization of all four plots
p1 + p2 + p3 + p4 +
  patchwork::plot_annotation(title = "Residuals of Popular Movies",
                             tag_levels = "A")

#### Data Sparsity -------------------------------------------------------------

# Calculating SVD on a large dataset with a lot of missing values is difficult.
# To visualize what is happening at the data-level, take the movie IDs from the 
# plots above and create a small subset.

long_sample <- test_set |> 
  filter(movieId %in% c(2959, 260, 1210, 25, 1704, 2959, 924, 1208)) |> 
  select(userId, movieId, rating) |> 
  slice_sample(n = 10) |> 
  remove_rownames()

# Matrix view of long-form data
# This format is not the correct format for SVD calculations
long_sample |> 
  as.matrix()

# To create a user-item matrix, we need to transform the matrix such that
# the movie IDs are the columns and the user IDs are the rows.
user_item_sample_matrix <- long_sample |> 
  pivot_wider(id_cols = userId,
              names_from = movieId,
              values_from = rating) |> 
  column_to_rownames("userId") |> 
  as.matrix()

# Notice that even in this small subset of the data using the most popular
# movies, there is a high rate of missing data
user_item_sample_matrix

# How much missing data is there in the entire dataset?

# Total number of cells in the full edx dataset
total_cells <- n_distinct(edx$userId) * n_distinct(edx$movieId)
total_cells

# Number of cells with ratings
valid_cells <- nrow(edx)
valid_cells

# Proportion of non-missing values
sparsity <- valid_cells/total_cells
sparsity

# Given the sparsity, we will use the irlba package to perform a truncated SVD
# approximating k singular vectors instead of the full number of predictors
# (movies), which in this case is over 10,000.

### Preprocessing --------------------------------------------------------------

# Before calculating the truncated SVD, we need to factorize the movie and user
# ID variables to ensure that both test and training sets include all the unique
# IDs from both columns and that the factors are listed in the same order

# Ensure both sets of IDs are unique are sorted in the same order
user_levels  <- sort(unique(train_set$userId))
movie_levels <- sort(unique(train_set$movieId))

# Add factorized version of IDs as integers and then match them according to 
# their row/column position of the original ID variables
train_set <- train_set |> 
  mutate(userId_x = as.integer(factor(userId)),
         movieId_x = as.integer(factor(movieId))) |> 
  mutate(userId_x = match(userId, user_levels),
         movieId_x = match(movieId, movie_levels))

# Perform the same preprocessing on the test set
test_set <- test_set |> 
  mutate(userId_x = as.integer(factor(userId)),
         movieId_x = as.integer(factor(movieId))) |> 
  mutate(userId_x = match(userId, user_levels),
         movieId_x = match(movieId, movie_levels))

# Save both datasets for later use
saveRDS(train_set, "9_Capstone/data/train-set.rds")
saveRDS(test_set,  "9_Capstone/data/test-set.rds")

### Model Testing --------------------------------------------------------------
# Singular value decomposition model that removes movie effect
# To save memory, switch to data.table syntax
setDT(train_set)
setDT(test_set)

# Calculate centered movie ratings
col_means <- train_set[, .(mean_ratings = mean(rating)), by = movieId]
train_set[col_means, centered := rating - mean_ratings, on = "movieId"]

# Create a sparse Matrix object using the Matrix package
y <- sparseMatrix(
  i = as.integer(factor(train_set$userId)),
  j = as.integer(factor(train_set$movieId)),
  x = train_set$centered,
  dimnames = list(
    user = levels(factor(train_set$userId)), 
    movie = levels(factor(train_set$movieId)))
)

# Run SVD function using the irlba package

t1 <- Sys.time() # Processing time should take less than one minute

# Calculate singular values using truncated number of singular vectors
# In this case, we select k = 50 as a starting point
s <- irlba(y, nv = 50, nu = 50)

# Calculate the predicted matrix according to the SVD formula
pred_mat <- s$u %*% diag(s$d) %*% t(s$v)

# Extract the predicted centered values based upon the user's row position and
# movie's column position in the test set
pred_centered <- pred_mat[cbind(test_set$userId_x, test_set$movieId_x)]

# Extract the column means for the test set's movies based upon the movie's
# positioning
movie_means <- col_means$mean_ratings[match(test_set$movieId,col_means$movieId)]

# Calculate the predicted values by adding the predicted centered values and
# movie means together
pred <- pred_centered + movie_means

Sys.time() - t1

# Calculate the RMSE of the predicted values vs the test set ratings
fit4_rmse <- sqrt(mean((pred - test_set$rating)^2))
fit4_rmse

## 5. SVD Movie + User Effects -------------------------------------------------
# The above model only removed the movie effects. To remove both movie and user 
# biases from the model, we need to calculate the movie means as before and then 
# use the b_i term to calculate the user means.

# Global average
mu_hat <- mean(train_set$rating)

# Dataset of movie means
movie_means <- train_set[, .(b_i = mean(rating - mu_hat)), by = movieId]

# Join movie_means with training set and calculate user means
user_means <- train_set[movie_means, on = "movieId"][
  , .(b_u = mean(rating - mu_hat - b_i)), by = userId]

# Since what will be left after removing user and movie biases will be residuals,
# create a copy of the training set with b_i and b_u terms added. Then, calculate
# residuals by subtracting global mean, b_i, and b_u term from each rating
train_resid <- train_set
train_resid[movie_means, b_i := i.b_i, on = "movieId"]
train_resid[user_means, b_u := i.b_u, on = "userId"]
train_resid[, residual := rating - mu_hat - b_i - b_u]

# Create sparse matrix object of the residuals. This is the matrix we'll use to
# calculate the SVD

y <- sparseMatrix(
  i = train_resid$userId_x,
  j = train_resid$movieId_x,
  x = train_resid$residual,
  dimnames = list(
    user = levels(factor(train_resid$userId)),
    movie = levels(factor(train_resid$movieId))
  ))

# Track processing time (should take less than 2 minutes)
t1 <- Sys.time()

# Estimate singular values using irbla() function, selecting k = 50 as arbitrary
# number of right and left singular vectors to estimate.
s <- irlba(y, nv = 50, nu = 50)

# Calculate predicted matrix according to SVD formula
pred_mat <- s$u %*% diag(s$d) %*% t(s$v)

# Extract residuals using the row/column positioning of the test set's user and 
# movie ID
pred_residuals <- pred_mat[cbind(test_set$userId_x, test_set$movieId_x)]

# Create vector for movie effects based upon movie ID position in test set
test_b_i <- movie_means$b_i[match(test_set$movieId, movie_means$movieId)]

# Create vector for user effects based upon user ID position in test set
test_b_u <- user_means$b_u[match(test_set$userId, user_means$userId)]

# Calculate predicted values, restricting values to 0.5 to 5.0 range.
pred <- clamp(mu_hat + test_b_i + test_b_u + pred_residuals)

Sys.time() - t1

# Calculate RMSE for SVD movie + user effects model
fit5_rmse <- RMSE(pred, test_set$rating)

fit5_rmse

## 6. Regularized SVD Movie & User Effects -------------------------------------
# To determine whether model could be improved with regularization, we need to
# use cross-validation on lambda and k values.

# Possible lambda values to test
lambdas <- seq(0, 5, 0.50)

# Possible k values to test. To reduce processing time, only two values are
# tested.
k_values <- c(50, 100)

# Use expand.grid() function to create a data frame from all combinations of
# lambda and k values - results in 22 different combinations.
grid <- expand.grid(lambda = lambdas, k = k_values)
grid

# Save global mean if not already saved
mu_hat <- mean(train_set$rating)

# Use doParallel package to create two copies of R running in parallel
# These copies are called "clusters," and need to communication with each
# other. Use the registerDoParallel() function to ensure communication between
# the two clusters
nc <- 2
cl <- makeCluster(nc)
registerDoParallel(cl)

# Explore important objects from R environment, including necessary libraries,
# train set, test set, and global mean.
clusterEvalQ(cl, {
  library(data.table)
  library(Matrix)
  library(irlba)
})

# Exporting data sets to two clusters may take a few minutes.
clusterExport(cl, c("train_set", "test_set", "mu_hat"))

# To calculate RMSE values of the 22 different models, write a function that
# takes in lambda and k values as arguments and returns the RMSE for each model
rmses <- function(lambda, k){
  # Calculate the b_i movie effects with an added penalty term, lambda, to mean 
  # formula
  movie_means_reg <- train_set[, .(b_i = sum(rating - mu_hat)/(.N + lambda)), by = movieId_x]
  
  # Calculate b_u user effects with penalty term lambda added to mean formula
  user_means_reg <- train_set[movie_means_reg, on = "movieId_x"][
    , .(b_u = sum(rating - mu_hat - b_i)/(.N + lambda)), by = userId_x]
  
  # Copy training set into new dataset
  train_resid <- train_set
  
  # Join regularized movie and user means to dataset
  train_resid[movie_means_reg, b_i := i.b_i, on = "movieId_x"]
  train_resid[user_means_reg, b_u := i.b_u, on = "userId_x"]
  
  # Create residual column for SVD calculations
  train_resid[, residual:= rating - mu_hat - b_i - b_u]
  
  # Create sparse matrix based on users (rows), movies (columns), and residuals
  # (values)
  y <- sparseMatrix(
    i = train_resid$userId_x,
    j = train_resid$movieId_x,
    x = train_resid$residual
  )
  
  # Calculate approximate k singular values and singular vectors
  s <- irlba(y, nv = k, nu = k)
  
  # Calculate predicted residuals according to SVD formula
  pred_residuals <- s$u %*% diag(s$d) %*% t(s$v)
  
  # Extract predicted residuals based upon users' and movies' positioning in
  # test set
  test_residuals <- pred_residuals[cbind(test_set$userId_x, test_set$movieId_x)]
  
  # Create vector of regularized movie effects based upon movie ID position in
  # test set
  test_b_i <- movie_means_reg[.(test_set$movieId_x), b_i, on = "movieId_x"]
  
  # Create vector of regularized user effects based upon user ID position in
  # test set
  test_b_u <- user_means_reg[.(test_set$userId_x), b_u, on = "userId_x"]
  
  # Calculate predicted values, restricting values within 0.5 to 5.0
  pred <- clamp(mu_hat + test_b_i + test_b_u + test_residuals)
  
  # Return RMSE of model
  RMSE(pred, test_set$rating)
}

# Track processing time - this should take about 30 minutes
# **NOTE: This code will take a significant amount of processing power. Close all
## programs running in the background before attempting to run.**
t1 <- Sys.time()

# Use foreach package (attached from doParallel package) to evaluate in parallel
# the rmses() function. 
## Pass through the ith row of the lambda and k columns
## Use the .combine = c argument to create a vector of all the RMSE values
results <- foreach(i = 1:nrow(grid), .combine = c) %dopar% {
  rmses(grid$lambda[i], grid$k[i])
}

Sys.time() - t1

# Store the results of the 22 RMSEs into a new column in the grid data frame
grid$rmse <- results 

# Close created cluster
stopCluster(cl)
stopImplicitCluster()

# Plot the results
lambda_plot <- ggplot(grid |> mutate(k = factor(k))) +
  geom_point(aes(lambda, rmse, color = k))

lambda_plot

# Lambda of model with lowest RMSE
lambda_m6 <- grid[which.min(grid$rmse),"lambda"]
lambda_m6

# k value of model with lowest RMSE
k_m6 <- grid[which.min(grid$rmse),"k"]
k_m6

# Lowest RMSE of regularized SVD movie + user effects model
fit6_rmse <- min(grid$rmse)
fit6_rmse

# RESULTS ----------------------------------------------------------------------
# Now that we know the model that had the lowest RMSE, we need to test the model
# on the final_holdout_test data frame that was created at the beginning before
# model training occurred.

## Final Model Testing ---------------------------------------------------------
### Calculated Predicted Residuals ---------------------------------------------
# Because we did not save the needed matrices in the previous step (we only
# saved the RMSE values), we need to calculate the predicted residuals from the
# training set here

# Calculate the regularized movie effects using the lambda identified from 
# model 6 above
movie_means_reg <- train_set[, .(b_i = sum(rating - mu_hat)/(.N + lambda_m6)), 
                             by = movieId_x]

# Calculate the regularized user effects using the lambda identified above
user_means_reg <- train_set[movie_means_reg, on = "movieId_x"][
  , .(b_u = sum(rating - mu_hat - b_i)/(.N + lambda_m6)), by = userId_x]

# Copy training set into new data table
train_resid <- train_set

# Join movie and user means to data table and calculate residuals
train_resid[movie_means_reg, b_i := i.b_i, on = "movieId_x"]
train_resid[user_means_reg, b_u := i.b_u, on = "userId_x"]
train_resid[, residual:= rating - mu_hat - b_i - b_u]

# Create sparse matrix of rows (users), columns (movies), and residuals (values)
y <- sparseMatrix(
  i = train_resid$userId_x,
  j = train_resid$movieId_x,
  x = train_resid$residual
)

# Calculate k singular values and singular vectors that were identified from
# model 6 above
s <- irlba(y, nv = k_m6, nu = k_m6)

# Calculate predicted residuals according to SVD formula
pred_residuals <- s$u %*% diag(s$d) %*% t(s$v)

### Testing Final Hold-Out Dataset ---------------------------------------------

# Load final_holdout_test dataset
load(file = "9_Capstone/data/final_holdout_test.rds")

# Factorize movie and user ID columns by first saving all unique IDs from train
# set
user_levels  <- sort(unique(train_set$userId))
movie_levels <- sort(unique(train_set$movieId))

# Copy final dataset into new data frame with factorized ID columns just as was
# done to the training set earlier
final_test <- final_holdout_test |> 
  mutate(userId_x = as.integer(factor(userId)),
         movieId_x = as.integer(factor(movieId))) |> 
  mutate(userId_x = match(userId, user_levels),
         movieId_x = match(movieId, movie_levels))

# Save dataframe as data.table object
setDT(final_test)

# Extract predicted residuals from test set based upon row/column position
test_residuals <- pred_residuals[cbind(final_test$userId_x, final_test$movieId_x)]

# Create a vector for movie effects based on movie ID's position in test dataset
test_b_i <- movie_means_reg[.(final_test$movieId_x), b_i, on = "movieId_x"]

# Create a vector for user effects based on user ID's position in test dataset
test_b_u <- user_means_reg[.(final_test$userId_x), b_u, on = "userId_x"]

# Calculate predicted values, restricting values within 0.5 to 5.0
pred <- clamp(mu_hat + test_b_i + test_b_u + test_residuals)

# Calculate RMSE of predicted ratings vs actual ratings
final_rmse <- RMSE(pred, final_test$rating)
final_rmse


## Comparing Model Accuracy ----------------------------------------------------

# Create data frame of RMSE from 6 training models and final holdout model
rmse_df <- data.frame("Model" = c("Average (Naive)",
                                  "Movie effect",
                                  "Movie + user effects",
                                  "Regularized movie + user effect",
                                  "SVD movie effect",
                                  "SVD movie + user effects",
                                  "Regularized SVD movie + user effects",
                                  "Final model"),
                      RMSE = c(naive_rmse,
                               fit1_rmse,
                               fit2_rmse,
                               fit3_rmse,
                               fit4_rmse,
                               fit5_rmse,
                               fit6_rmse,
                               final_rmse),
                      test.data = c("edX Test Set",
                                    "edX Test Set",
                                    "edX Test Set",
                                    "edX Test Set",
                                    "edX Test Set",
                                    "edX Test Set",
                                    "edX Test Set",
                                    "Final Hold-out Test Set")
) |> 
  mutate(across(ends_with("rmse"), ~ round(., 4)))

# Create bar plot of final results sorted by size of RMSE
# Notice how regularization provided only minimal improvement for SVD movie +
# user model. The results show that regularization wasn't necessary since over-
# fitting wasn't a problem in the non-regularized SVD movie + user model.

rmse_df |> 
  ggplot(aes(RMSE, reorder(Model, RMSE))) +
  geom_col(aes(fill = test.data)) +
  geom_vline(xintercept = 0.8643,
             linetype = "dashed",
             color = "#1F968BFF",
             linewidth = 1) +
  geom_label(aes(label = RMSE,
                 fill = test.data),
             color = ifelse(rmse_df$test.data == "edX Test Set", "white", "black"),
             hjust = 1,
             nudge_x = -0.01,
             fontface = "bold",
             linewidth = NA,
             show.legend = FALSE) +
  geom_label(data = data.frame(x = 0.8712, 
                               y = 9,
                               label = "RMSE = 0.8712\nBell et al.'s (2008) winning algorithm"),
             aes(x = x, y = y, label = label),
             nudge_x = -0.1,
             color = "#1F968BFF",
             fontface = "bold",
             linewidth = NA,
             hjust = 0.5) +
  labs(y = NULL,
       x = NULL,
       fill = NULL,
       title = "Root Mean Square Error (RMSE) Values of Tested Models on MovieLens Dataset") +
  scale_fill_manual(values = c("#440154FF", "#FDE725FF")) +
  expand_limits(y = c(1, 9.5)) +
  theme_minimal() +
  theme(legend.position = "top",
        plot.title.position = "plot")


