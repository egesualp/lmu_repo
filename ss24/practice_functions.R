# Practising R - Set 1
# lapply ~ for loop

result_list <- list()
result_list <- lapply(3:33, function(x) x^2)
result_list

result_list <- lapply(1:10, function(x) paste('id: ', x))
