#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  #
#  Sara E. Brady, Ph.D.                            #
#  HarvardX Data Science Professional Certificate  #
#  Capstone Personal Project                       #
#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  #

# Libraries --------------------------------------------------------------------
if (!require(tidyverse)) install.packages('tidyverse')
if (!require(caret)) install.packages('caret')
if (!require(caretEnsemble)) install.packages('caretEnsemble')
if (!require(pROC)) install.packages('pROC')
if (!require(kableExtra)) install.packages('kableExtra')
if (!require(networkD3)) install.packages('networkD3')
if (!require(curl)) install.packages('curl')
if (!require(ggrepel)) install.packages('ggrepel')

options(scipen = 999)

# Load Data --------------------------------------------------------------------
data_url <- "https://raw.githubusercontent.com/bradyse/HarvardX/main/personal-submission-files/leg_labels.rds"

leg_data <- "leg_labels.rds"
if(!file.exists(leg_data)) download.file(data_url, leg_data, method = "curl", mode = "wb")

leg_labels <- readRDS(file = "leg_labels.rds")

## ---- Tables 1 & 2 -----------------------------------------------------------
str(leg_labels)

# Exploratory Data Analysis ----------------------------------------------------
## ---- Fig. 1. Sankey Diagram -------------------------------------------------
df_sankey <- leg_labels |> 
  mutate(Cmte.Assign = replace_when(Committee, 
                                    Committee == "None (A Bills)" ~ "A Bill",
                                    Committee == "None" ~ "None (Appropriation Bills)",
                                    Committee == "None (Withdrawn Bills)" ~ "Bill Withdrawn Before Committee Assignment"),
         GenFile.Outcome = case_when(Placed.GenFile == "Placed on General File" ~ "Placed on General File",
                                     Placed.GenFile == "Was Not Placed on General File" 
                                     & Bill.Withdrawn == "Bill Not Withdrawn" 
                                     & Placed.SelectFile == "Was Not Placed on Select File" ~ "Held in Committee",
                                     Committee != "None (Withdrawn Bills)"
                                     & Placed.SelectFile == "Was Not Placed on Select File"
                                     & Bill.Withdrawn == "Bill Withdrawn" ~ "Bill Withdrawn Before General File"),
         SelFile.Outcome = case_when(Placed.SelectFile == "Placed on Select File" ~ "Placed on Select File",
                                     Placed.GenFile == "Placed on General File"
                                     & Placed.SelectFile == "Was Not Placed on Select File"
                                     & Bill.Withdrawn == "Bill Not Withdrawn" ~ "Remain on General File",
                                     Placed.GenFile == "Placed on General File"
                                     & Placed.SelectFile == "Was Not Placed on Select File"
                                     & Bill.Withdrawn == "Bill Withdrawn" ~ "Bill Withdrawn Before Select File",
                                     GenFile.Outcome == "Held in Committee"
                                     & Indefinitely.Postponed == "Indefinitely Postponed" ~ "Indefinitely Postponed"),
         FReadFile.Outcome = case_when(Placed.FinalRead == "Placed on Final Reading" ~ "Placed on Final Reading",
                                       Placed.SelectFile == "Placed on Select File"
                                       & Placed.FinalRead == "Was Not Placed on Final Reading" ~ "Remain on Select File",
                                       SelFile.Outcome == "Remain on General File"
                                       & Indefinitely.Postponed == "Indefinitely Postponed" ~ "Indefinitely Postponed"),
         Gov.Outcome = case_when(Approved.Gov == "Approved by Governor" ~ "Approved by Governor",
                                 Approved.Gov == "Was Not Approved by Governor" & Bill.Outcome == "Passed" ~ "Bill Passed Without Governor",
                                 Bill.Outcome == "Vetoed & No Override" ~ "Vetoed by Governor",
                                 Bill.Outcome == "Failed on Final Reading" ~ "Failed on Final Reading",
                                 Placed.FinalRead == "Placed on Final Reading"
                                 & Bill.Outcome == "Indefinitely Postponed" ~ "Remain on Final Reading",
                                 FReadFile.Outcome == "Remain on Select File"
                                 & Indefinitely.Postponed == "Indefinitely Postponed" ~ "Indefinitely Postponed"),
         Final.Outcome = case_when(Gov.Outcome == "Approved by Governor" ~ "Bill Becomes Law",
                                   Gov.Outcome == "Bill Passed Without Governor" ~ "Bill Becomes Law",
                                   Gov.Outcome == "Remain on Final Reading"
                                   & Indefinitely.Postponed == "Indefinitely Postponed" ~ "Indefinitely Postponed")) |> 
  select(1:6, Passed, Indefinitely.Postponed, Bill.Withdrawn, Bill.Outcome,
         Cmte.Assign, GenFile.Outcome, SelFile.Outcome, FReadFile.Outcome,
         Gov.Outcome, Final.Outcome) |> 
  mutate(across(7:16, ~factor(.))) |> 
  group_by(Cmte.Assign) |> 
  add_count(Cmte.Assign, name = "n.Cmte.Assign") |> 
  ungroup() |> 
  mutate(Cmte.Assign = fct_reorder(Cmte.Assign, -n.Cmte.Assign))

node_names <- c(
  "Introduced",                               
  levels(df_sankey$Cmte.Assign),
  "Placed on General File",
  "Held in Committee",
  "Bill Withdrawn Before General File",
  "Placed on Select File", 
  "Remain on General File",
  "Bill Withdrawn Before Select File",
  "Placed on Final Reading",
  "Remain on Select File",
  "Approved by Governor",
  "Bill Passed Without Governor",
  "Failed on Final Reading",
  "Vetoed by Governor",
  "Remain on Final Reading",
  "Bill Becomes Law",
  "Indefinitely Postponed"
)

nodes <- data.frame(name = node_names,
                    stringsAsFactors = FALSE)

node_id_fn <- function(node) match(node, nodes$name) - 1L

node_labels <- cbind(nodes, data.frame(source = node_id_fn(node_names))) |> 
  rename(source.label = name) |> 
  mutate(target.label = source.label,
         target = source)

# Introduced -> Committee category
intro_links <- df_sankey |> 
  mutate(entry_node = Cmte.Assign) |> 
  count(entry_node) |> 
  arrange(-n) |> 
  mutate(source = node_id_fn("Introduced"),
         target = node_id_fn(entry_node),
         value = n) |> 
  select(source, target, value)

# Committee and A Bills -> General File Outcome
cmte_links <- df_sankey |> 
  mutate(entry_node = Cmte.Assign,
         outcome_node = GenFile.Outcome) |> 
  count(entry_node, outcome_node) |> 
  mutate(source = node_id_fn(entry_node),
         target = node_id_fn(outcome_node),
         value = n) |> 
  filter(!is.na(outcome_node)) |> 
  select(source, target, value)

# General File -> Select File
genf_links <- df_sankey |> 
  mutate(entry_node = GenFile.Outcome,
         outcome_node = SelFile.Outcome)  |>
  count(entry_node, outcome_node) |> 
  mutate(source = node_id_fn(entry_node),
         target = node_id_fn(outcome_node),
         value = n) |> 
  filter(!is.na(outcome_node)) |> 
  select(source, target, value)

# Select File -> Final Reading
self_links <- df_sankey |> 
  mutate(entry_node = SelFile.Outcome,
         outcome_node = FReadFile.Outcome) |> 
  count(entry_node, outcome_node) |> 
  mutate(source = node_id_fn(entry_node),
         target = node_id_fn(outcome_node),
         value = n) |> 
  filter(!is.na(outcome_node)) |> 
  select(source, target, value)

# Final Reading -> Governor Outcome
finread_links <- df_sankey |> 
  mutate(entry_node = FReadFile.Outcome,
         outcome_node = Gov.Outcome) |> 
  count(entry_node, outcome_node) |> 
  mutate(source = node_id_fn(entry_node),
         target = node_id_fn(outcome_node),
         value = n) |> 
  filter(!is.na(outcome_node)) |> 
  select(source, target, value)

# Governor -> Final Outcome
gov_links <- df_sankey |> 
  mutate(entry_node = Gov.Outcome,
         outcome_node = Final.Outcome) |> 
  count(entry_node, outcome_node) |> 
  mutate(source = node_id_fn(entry_node),
         target = node_id_fn(outcome_node),
         value = n) |> 
  na.omit() |> 
  select(source, target, value)

links <- rbind(intro_links, cmte_links, genf_links,
               self_links, finread_links, gov_links
) |> 
  left_join(node_labels[,1:2], by = "source") |> 
  left_join(node_labels[,3:4], by = "target")

sankeyNetwork(
  Links  = links,
  Nodes  = nodes,
  Source = "source",
  Target = "target",
  Value  = "value",
  NodeID = "name",
  fontSize = 10,
  nodeWidth = 10,
  sinksRight = FALSE,
  iterations = 0,
  margin = list(top = 0, right = 0, bottom = 0, left = 0)
)

## ---- Fig. 2. Bills by Outcome per Session -----------------------------------

bills_by_outcome_per_session <- leg_labels |> 
  group_by(Leg.Years, Bill.Outcome) |> 
  count()

bills_by_outcome_per_session |> 
  ggplot(aes(y = fct_rev(Leg.Years), x = n, fill = str_wrap(Bill.Outcome, 20))) +
  geom_col() +
  geom_text(aes(x = n, y = Leg.Years, 
                label = if_else(n > 100, format(n, big.mark = ","),"")),
            position = position_stack(vjust = 0.5),
            size = 6,
            fontface = "bold",
            color = "white") +
  labs(x = NULL,
       y = NULL,
       fill = NULL,
       title = "Number of Bills per Legislative Session by Bill Outcome") +
  theme_minimal() +
  theme(legend.position = "top",
        plot.title.position = "plot",
        plot.title = element_text(size = 16),
        plot.subtitle = element_text(size = 16),
        axis.text = element_text(face = "bold", size = 12),
        legend.text = element_text(size = 12)) +
  scale_fill_brewer(palette = "Paired") 

## ---- Table. 3. Pass Rate (Including Rider Bills) by Characteristic ----------
build_all_bills_pass_rate_df <- function(var){
  leg_labels |> 
    group_by(.data[[var]]) |> 
    count(Passed.All.Bills) |> 
    mutate(`Percentage of Passed Bills` = round(n/sum(n)*100, 0)) |> 
    filter(str_detect(Passed.All.Bills, "Passed")) |>
    select(-c(Passed.All.Bills, n)) |>
    rename(Variable = 1) |>
    ungroup() |>
    mutate(Variable = str_replace(Variable, "\n", " "))
}

pass_rate_df_3 <- build_all_bills_pass_rate_df("Leg.Years") |> 
  rbind(build_all_bills_pass_rate_df("Introduced.Biennium")) |> 
  rbind(build_all_bills_pass_rate_df("Introducer.Type")) |> 
  rbind(build_all_bills_pass_rate_df("IsAmendedIntoPackageBill")) |> 
  rbind(build_all_bills_pass_rate_df("IsPackageBill")) |> 
  rbind(build_all_bills_pass_rate_df("Bill.Package.Category")) |> 
  rbind(build_all_bills_pass_rate_df("IsCarryoverBill")) |> 
  rbind(build_all_bills_pass_rate_df("IsPriority")) |> 
  rbind(build_all_bills_pass_rate_df("Priority.Owner")) |> 
  rbind(build_all_bills_pass_rate_df("Is.Term.Limit")) |> 
  rbind(build_all_bills_pass_rate_df("IsABill")) |> 
  rbind(build_all_bills_pass_rate_df("HasABill")) |> 
  rbind(build_all_bills_pass_rate_df("AmendedInto.HasABill")) |> 
  rbind(build_all_bills_pass_rate_df("AmendedInto.Passed"))

kable(pass_rate_df_3) |> 
  pack_rows(index = c("Legislative Years" = 9, 
                      "Introduced Biennium" = 3,
                      "Introducer Category" = 3,
                      "Amended Into Package Bill" = 2,
                      "Package Bill" = 2,
                      "Package Bill Category" = 3,
                      "Carryover Bill" = 2,
                      "Priority Bill" = 2,
                      "Priority Owner" = 4,
                      "Introduced by Term-Limited Senator" = 2,
                      "A-Bill" = 2,
                      "Has A-Bill" = 2,
                      "Amended-Into Bill (Provisions) Has an A Bill" = 2,
                      "Amended-Into Bill (Provisions) was Passed" = 2
  )) |> 
  column_spec(2, color = "white",
              background = spec_color(pass_rate_df_3$`Percentage of Passed Bills`))

## ---- Fig. 3. Names Added to Introduced Bills --------------------------------
leg_names <- leg_labels |> 
  select(Leg.Years, LB, Num.NameAdded) 

leg_names |> 
  filter(Num.NameAdded > 0) |> 
  ggplot(aes(y = fct_rev(Leg.Years), x = Num.NameAdded)) +
  geom_boxplot() +
  geom_jitter(width = 0, height = 0.2, alpha = 0.3) +
  labs(x = NULL,
       y = NULL,
       title = "Frequency of Names Added to Introduced Bills with at Least One Name Added") +
  theme_minimal() +
  theme(axis.text = element_text(face = "bold", size = 10),
        legend.text = element_text(size = 10),
        plot.title.position = "plot")
## ---- Fig. 4. Number of Bills Amended Into Package Bills ---------------------
leg_packages <- leg_labels |> 
  filter(IsPackageBill == "Is a Package Bill") |> 
  select(Leg.Years, LB, Number.Bills.in.Package) 

leg_packages |> 
  ggplot(aes(y = fct_rev(Leg.Years), x = Number.Bills.in.Package)) +
  geom_boxplot() +
  geom_jitter(width = 0, height = 0.2, alpha = 0.3) +
  labs(x = NULL,
       y = NULL,
       title = "Number of Bills Amended Into Package Bills") +
  theme_minimal() +
  theme(axis.text = element_text(face = "bold", size = 10),
        legend.text = element_text(size = 10),
        plot.title.position = "plot")
## ---- Fig. 5. Pass Rate by Number of Bills Added -----------------------------
# Excludes bills with zero names added

leg_labels |> 
  filter(Num.NameAdded > 0) |> 
  ggplot(aes(Num.NameAdded, fill = str_wrap(Passed, 70))) +
  geom_histogram(binwidth = 1) +
  labs(x = NULL,
       fill = NULL,
       title = "Pass Rate Distribution of the Number of Names Added to a Bill") +
  theme_minimal() +
  theme(legend.position = "top",
        axis.text = element_text(face = "bold", size = 10),
        legend.text = element_text(size = 10),
        plot.title.position = "plot") +
  scale_fill_brewer(palette = "Paired")

## ---- Fig. 6. Number of Bills Amended into Bill Package ----------------------
# Excludes bills that are not package bills
leg_labels |> 
  filter(Number.Bills.in.Package > 0) |> 
  ggplot(aes(Number.Bills.in.Package, fill = str_wrap(Passed, 70))) +
  geom_histogram(binwidth = 1) +
  labs(x = NULL,
       fill = NULL,
       title = "Pass Rate Distribution of Number of Bills Amended into Package Bills") +
  theme_minimal() +
  theme(legend.position = "top",
        axis.text = element_text(face = "bold", size = 10),
        legend.text = element_text(size = 10),
        plot.title.position = "plot") +
  scale_fill_brewer(palette = "Paired")

# Final Training and Testing Datasets ------------------------------------------
## ---- Train and Test Data Sets -----------------------------------------------
# Final variables selected for model testing
leg_final <- leg_labels |> 
  filter(Bill.Withdrawn != "Bill Withdrawn") |>
  mutate(across(where(is.character), ~factor(.))) |> 
  select(Passed.All, IsCarryoverBill, Is.Term.Limit, HasABill, Placed.GenFile,
         Introduced.Biennium, Introducer.Type, Bill.Package.Category,
         Priority.Owner, Committee, Leg, Num.NameAdded)

leg_main <- leg_final |> 
  filter(Leg != 108)

leg_final_test <- leg_final |> 
  filter(Leg == 108)

## ---- Table 4. Legislative Variables Selected for Model Testing --------------
str(leg_final)

# Preprocessing ---------------------------------------------------------------
## ---- Dummy coding -----------------------------------------------------------
predictors <- select(leg_main, -Passed.All)
dummies <- dummyVars(~ ., data = predictors, fullRank = TRUE)
leg_dummy <- data.frame(predict(dummies, newdata = predictors))

leg_dummy$Passed.All <- leg_main$Passed.All
str(leg_dummy)

## ---- Near-zero variance -----------------------------------------------------
nzv <- nearZeroVar(leg_dummy, saveMetrics = TRUE)
nzv[,][nzv$nzv,]

## ---- Recoding infrequent subcategories ------------------------------------------------
leg_main_final <- leg_main |> 
  filter(Introduced.Biennium != "Special",
         Committee != "Redistricting",
         Committee != "None") |> 
  mutate(Cmte = factor(case_when(Committee == "Business and Labor" ~ "Other",
                                 Committee == "Executive Board" ~ "Other",
                                 Committee == "General Affairs" ~ "Other",
                                 Committee == "Natural Resources" ~ "Other",
                                 Committee == "Nebraska Retirement Systems" ~ "Other",
                                 Committee == "Urban Affairs" ~ "Other",
                                 TRUE ~ Committee)),
         Introducer = factor(if_else(
           Introducer.Type == "Senator", "Senator", "Committee or Speaker")),
         Is.Standalone = factor(if_else(Bill.Package.Category == "Standalone Bill", "Standalone Bill", "Package Bill or Provisions Amended")),
         Is.Priority = factor(if_else(Priority.Owner == "Is Not a Priority Bill", "Not a Priority Bill", "Is a Priority Bill")))|> 
  select(-c(Committee, Introducer.Type, Bill.Package.Category, Priority.Owner)) |> 
  mutate(across(where(is.factor), ~fct_drop(.)))

## ---- Dummy coding following subcategory recode ------------------------------
predictors <- select(leg_main_final, -Passed.All)
dummies <- dummyVars(~ ., data = predictors, fullRank = TRUE)
leg_dummy <- data.frame(predict(dummies, newdata = predictors))

leg_dummy$Passed.All <- leg_main_final$Passed.All
str(leg_dummy)

### ---- Near-zero variance ----------------------------------------------------
nzv <- nearZeroVar(leg_dummy, saveMetrics = TRUE)
nzv[,][nzv$nzv,]

### ---- High correlations -----------------------------------------------------
data_cor <- cor(leg_dummy |> 
                  mutate(Passed.All = if_else(Passed.All == "Yes", 1, 0))
)
findCorrelation(data_cor, cutoff = 0.75)

### ---- Linear combinations ---------------------------------------------------
comboInfo <- findLinearCombos(leg_dummy |> 
                                mutate(Passed.All = if_else(Passed.All == "Yes", 1, 0))
)
comboInfo

### ---- Final preprocessed data set -------------------------------------------
leg_dummy <- leg_dummy |> 
  mutate(across(1:5, ~factor(.)),
         across(8:20, ~factor(.))) 

leg_preproc <- preProcess(leg_dummy, method = c("scale", "center", "nzv"))
leg_preproc

leg_transf <- predict(leg_preproc, leg_dummy)

# Training ---------------------------------------------------------------------
## ---- Partition data ---------------------------------------------------------
y <- leg_transf$Passed.All

set.seed(2013)
index <- createDataPartition(y, times = 1, p = 0.75, list = FALSE)
train_set <- leg_transf[index,]
test_set <- leg_transf[-index,]

## ---- Resampling method ------------------------------------------------------
train_control <- trainControl(method = "repeatedcv",
                              repeats = 3,
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)


## ---- Testing Multiple Models ------------------------------------------------
algorithms <- c("knn", "glm", "glmnet", "rf", "rpart")

lapply(algorithms, modelLookup) |> bind_rows()

tuning_grids <- list(
  knn = caretModelSpec(
    method = "knn",
    tuneGrid = data.frame(k = seq(3, 21, 2))),
  glmnet = caretModelSpec(
    method = "glmnet",
    tuneGrid = expand.grid(
      alpha = 0:1, # alpha == 0 (ridge) alpha == 1 (LASSO)
      lambda = seq(0.0001, 0.1, length = 10))), # regularization
  rf = caretModelSpec(
    method = "rf",
    tuneGrid = data.frame(mtry = ceiling(seq(2, 19, length = 10)))),
  rpart = caretModelSpec(
    method = "rpart",
    tuneGrid = data.frame(cp = seq(0, 0.05, len = 10))
  )
)

# NOTE THE CODE BELOW TAKES ABOUT 20 MINUTES TO RUN. IF YOU DO NOT WANT TO WAIT
# THAT LONG,YOU CAN DOWNLOAD THE MODELS DATA OBJECT USING THIS CODE:
#
models_url <- "https://raw.githubusercontent.com/bradyse/HarvardX/main/personal-submission-files/leg_models.rds"
models_data <- "leg_models.rds"
if(!file.exists(models_data))
  download.file(models_url, models_data, method = "curl", mode = "wb")
models <- readRDS(file = "leg_models.rds")

# set.seed(2013)
# t1 <- Sys.time()
# models <- caretList(Passed.All ~ .,
#                     methodList = c("glm"),
#                     data = train_set,
#                     trControl = train_control,
#                     tuneList = tuning_grids)
# t2 <- Sys.time()
# 
# t2 - t1

### ---- Training model comparison ---------------------------------------------
all_models_results <- resamples(models)
summary(all_models_results)

# Identifies whether glmnet selected ridge (alpha = 0) or LASSO (alpha = 1)
models$glmnet$bestTune

summary(diff(all_models_results))


### ---- Fig. 7 Boxplot of ROC, Sensitivity, and Specificity by Model ----------
# Create box plot to visualize model performance
bwplot(all_models_results, scales = list(x = list(relation = "free"),
                                         y = list(relation = "free")))

# Construct correlation matrix and scatterplot
modelCor(all_models_results)

### ----- Fig. 8 Scatterplot of ROC Values by Model ----------------------------
splom(all_models_results)

## ---- Evaluation on Test Data Set --------------------------------------------
### ---- Confusion matrices ----------------------------------------------------

# Predicted values
y_hats <- lapply(models, function(x) predict(x, test_set))

cmats <- lapply(y_hats, function(x) {
  confusionMatrix(x, test_set$Passed.All, positive = "Yes")
})

cmats_accuracy <- sapply(cmats, function(x) x$overall["Accuracy"])
cmats_sensitivity <- sapply(cmats, function(x) x$byClass["Sensitivity"])
cmats_specificity <- sapply(cmats, function(x) x$byClass["Specificity"])

# Create table of all model summary statistics
tibble(Model = names(cmats),
       Accuracy = cmats_accuracy,
       Sensitivity = cmats_sensitivity,
       Specificity = cmats_specificity) |> 
  arrange(-Accuracy)

# Confusion table for GLM
cmats$glm$table

### ---- ROC curves ------------------------------------------------------------

#### ---- Plot ROC curves ------------------------------------------------------
# Predicted probabilities
probs <- lapply(models, function(x) predict(x, test_set,
                                            type = "prob")$Yes)

# ROC curve at each predicted probability
rocs <- lapply(probs, function(x) roc(test_set$Passed.All, x))

# Build ROC data frame
roc_data <- bind_rows(
  lapply(names(rocs), function(model) {
    roc_obj <- rocs[[model]]
    
    tibble(Model = model,
           FPR = 1 - roc_obj$specificities,
           TPR = roc_obj$sensitivities,
           AUC = str_extract(auc(roc_obj),
                             "\\d+.\\d{3}")
    )
  })
)

# ROC summary data frame
roc_summary_df <- roc_data |> 
  select(Model, AUC) |> 
  distinct() |> 
  mutate(Model_labels = paste0(Model, " AUC: ", AUC))

# ROC model labels used for plotting
roc_models <- unique(roc_data$Model)
roc_labels <- roc_summary_df$Model_labels
names(roc_labels) <- roc_models

## Fig. 9 ROC Curves by Model --------------------------------------------------
ggplot(roc_data,
                aes(x = FPR,
                    y = TPR,
                    color = Model)) +
  geom_line(linewidth = 1) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "grey") +
  scale_color_brewer(palette = "Paired",
                     labels = roc_labels) +
  labs(title = "ROC Curves by Model",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal() +
  theme(panel.grid = element_blank(),
        axis.line = element_line(color = "darkgrey"),
        axis.ticks = element_line(color = "darkgrey"),
        plot.title.position = "plot")

#### ---- DeLong ROC comparison test -------------------------------------------
pairs <- combn(roc_models, 2, simplify = FALSE)

roc_comparisons <- map_dfr(pairs, function(pair) {
  model_A <- pair[1]
  model_B <- pair[2]
  
  test_res <- roc.test(rocs[[model_A]], rocs[[model_B]])
  
  tibble(
    Model.Comparison = paste(model_A, "vs.", model_B),
    D.Statistic      = test_res$statistic,
    p.value.raw      = test_res$p.value
  )
})


roc_comparisons <- roc_comparisons %>%
  mutate(p.value.adj = p.adjust(p.value.raw, method = "bonferroni"))

print(roc_comparisons)

### ---- Variable importance ---------------------------------------------------
# Fitted GLM model
fit_glm <- train(Passed.All ~ ., method = "glm", data = train_set, trControl = train_control, metric = "ROC")

# Predicted values from GLM model
y_hat_glm <- predict(fit_glm, test_set)
probs_glm <- predict(fit_glm, test_set, type = "prob")$Yes

## Fig. 10. Variance Importance for GLM Model ----------------------------------
# Used `fit_glm` because `varImp(models$glm)` returns an error
plot(varImp(fit_glm))

# Plot average predicted probabilities for variables of high importance
df_plot <- data.frame(prob = probs_glm,
                      fit = as.numeric(y_hat_glm),
                      standalone = test_set$Is.Standalone.Standalone.Bill,
                      genfile = test_set$Placed.GenFile.Was.Not.Placed.on.General.File,
                      carryover = test_set$IsCarryoverBill.Is.Not.Carryover.Bill,
                      biennium = test_set$Introduced.Biennium.Second) |> 
  mutate(biennium = if_else(biennium == 1, "Second Regular Session", "First Regular Session"),
         carryover = if_else(carryover == 1, "Not a Carryover Bill", "Carryover Bill"),
         genfile = if_else(genfile == 1, "Not Placed on General File", "Placed on General File"),
         standalone = if_else(standalone == 1, "Is Standalone Bill", "Is Package or Rider Bill"),
  ) |> 
  pivot_longer(cols = c(standalone, carryover, genfile, biennium)) |> 
  group_by(name, value) |> 
  reframe(mean_prob = mean(prob))

## Fig. 11. Mean Predicted Probabilities in GLM Model --------------------------

df_plot |> 
  ggplot(aes(x = value, y = mean_prob, group = name)) +
  geom_line() +
  geom_point() +
  geom_text_repel(aes(label = round(mean_prob, 3)),
                  direction = "y") +
  facet_wrap(vars(name), scales = "free_x") +
  labs(x = NULL, y = NULL,
       title = "Mean Predicted Probabilities") +
  theme(plot.title.position = "plot")


# Results ---------------------------------------------------------------------
## Final Holdout Preprocessing ------------------------------------------------

# Recode variables
leg_final_test_selectvars <- leg_final_test |> 
  filter(Introduced.Biennium != "Special",
         Committee != "Redistricting",
         Committee != "None") |> 
  mutate(Cmte = factor(case_when(Committee == "Business and Labor" ~ "Other",
                                 Committee == "Executive Board" ~ "Other",
                                 Committee == "General Affairs" ~ "Other",
                                 Committee == "Natural Resources" ~ "Other",
                                 Committee == "Nebraska Retirement Systems" ~ "Other",
                                 Committee == "Urban Affairs" ~ "Other",
                                 TRUE ~ Committee)),
         Introducer = factor(if_else(
           Introducer.Type == "Senator", "Senator", "Committee or Speaker")),
         Is.Standalone = factor(if_else(Bill.Package.Category == "Standalone Bill", "Standalone Bill", "Package Bill or Provisions Amended")),
         Is.Priority = factor(if_else(Priority.Owner == "Is Not a Priority Bill", "Not a Priority Bill", "Is a Priority Bill")))|> 
  select(-c(Committee, Introducer.Type, Bill.Package.Category, Priority.Owner)) |> 
  mutate(across(where(is.factor), ~fct_drop(.)))

# Dummy code variables
predictors_test <- select(leg_final_test_selectvars, -Passed.All)
dummies_test <- dummyVars(~ ., data = predictors_test, fullRank = TRUE)
leg_dummy_test <- data.frame(predict(dummies_test, newdata = predictors_test)) |> 
  mutate(across(1:5, ~factor(.)),
         across(8:20, ~factor(.)))
leg_dummy_test$Passed.All <- leg_final_test_selectvars$Passed.All

# Center and scale variables
# Took "nzv" off because Leg = 108
leg_preproc_test <- preProcess(leg_dummy_test, method = c("scale", "center"))
leg_preproc

# Transform data set
leg_transf_test <- predict(leg_preproc_test, leg_dummy_test)

str(leg_transf_test)

## ---- Final Model Performance ------------------------------------------------
y_hat_glm_final <- predict(fit_glm, leg_transf_test)
probs_glm_final <- predict(fit_glm, leg_transf_test, type = "prob")$Yes

confusionMatrix(y_hat_glm_final, leg_transf_test$Passed.All, positive = "Yes")

roc_glm_final <- roc(leg_transf_test$Passed.All, probs_glm_final)

## ---- Fig. 12. ROC Curve in Final GLM Model ----------------------------------
plot(roc_glm_final, print.auc = TRUE)
