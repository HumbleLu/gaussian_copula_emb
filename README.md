# gaussian_copula_emb
code for implementing Gaussian Copula Embeddings (Lu, C., & Peltonen, J. (2022). Gaussian Copula Embeddings. Advances in Neural Information Processing Systems, 35, 22078-22089.)

When using the code, please do:

```{r, include = FALSE}
source("~/gaussian_copula_emb/utils.R")
```

to load necessary functions.

To create the input data, please see the example using Reddit hyperlink data set (example dataset provided)

```{r, include = FALSE}
load("subreddit_hypertext_processed.RData") # processed dataset

meta<- list(item_list = target_subreddit_map_dat$target_subreddit_id,
            context_list = source_subreddit_map_dat$source_subreddit_id,
            n_len = length(item_index),
            var_num = nrow(val_mtx),
            p_len = p_len)

item_len<- length(meta$item_list)
context_len<- length(meta$context_list)
n_len<- meta$n_len
var_num<- meta$var_num

data<- list(item_vec = item_index,
            val_mtx = val_mtx)
data$context_index = context_index
data$context_val = lapply(data$context_index, function(x) rep(1, length(x)))

meta$var_num<- nrow(data$val_mtx)

var_params<- list(phi = matrix(rnorm(item_len * p_len, 0, 0.01), p_len, item_len),
                  alpha = lapply(1:var_num, function(i) matrix(rnorm(context_len * p_len, 0, 0.01), context_len, p_len)))
```

Initialize and train a GCE model by

```{r, include = FALSE}
gce_model_init<- list(meta = meta, data = data, var_params = var_params)
gce_model_trained<- update_gce_model(gce_model_init, epochs = epochs, n_minibatches = n_minibatches,
                                     n_neg_samples = n_neg_samples, alpha_learn = alpha_learn, trace_indices = 1:500)

```
