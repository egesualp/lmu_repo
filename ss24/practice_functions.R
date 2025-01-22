# Practising R - Set 1
# lapply ~ for loop

result_list <- list()
result_list <- lapply(3:33, function(x) x^2)
result_list

result_list <- lapply(1:10, function(x) paste('id: ', x))

exp(1.63197) + 1.96*exp(1.63197)*0.05895
sqrt(exp(2 * 1.63197) / exp(1.63197))

# Aggregate
# Sample data frame
dt <- data.frame(
  fold = c(1, 1, 1, 2, 2, 2),
  group = c('A', 'A', 'B', 'A', 'A', 'B'),
  credit_risk = c('good', 'bad', 'good', 'bad', 'bad', 'good'),
  score = c(90, 85, 88, 92, 91, 87)
)

# One value to aggregate, by two group
aggregate(
  credit_risk ~ fold + group,
  data = dt,
  FUN = function(x) sum(x == "good") / sum(!is.na(x))
)

# Two value to aggregate
ag_1 <- aggregate(
  credit_risk ~ fold,
  data = dt,
  FUN = function(x) sum(x == "good") / sum(x == "bad")
)

ag_2 <- aggregate(
  score ~ fold,
  data = dt,
  mean
)

ag_f <- merge(ag_1, ag_2, by = "fold")
ag_f

aggregate(.~credit_risk, data=dt, mean)

## Testing basic filterings with data.table
dtest <- data.table(
  fold = c(1, 1, 1, 2, 2, 2),
  group = c('A', 'A', 'B', 'A', 'A', 'B'),
  credit_risk = c('good', 'bad', 'good', 'bad', 'bad', 'good'),
  score = c(90, 85, 88, 92, 91, 87)
)

dtest[score > 90,]
