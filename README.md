# Walmart Demand Forecasting
## time series analysis
- Kaggle: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data
- Notice: I study **winner's R code** and reproduce his code by python3
- Here is **his comment and original code**:
	- github: https://github.com/davidthaler/Walmart_competition_code
	- explanation: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8125#latest-357454 
	- key adjustment: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8028
- Models reproduced by python3
	- PCA + ETS(ExponenTial Smoothing)
	- PCA + stl + SARIMAX
	- +) PCA + customed seasonal adjust+ SARIMA: for seasonal adjustment,  I dedcut the average of every week which is considered deterministic seasonality (eg. avg(2010/2011/2012 1st week), ..., avg(2010/2011 52th week))

## Key concept in the Winner's models

1. PCA: PCA is used to find signal through the specific department. It additionally make it easier to deal with missing values which is NA or which is not exist in `train.csv`, but in `test.csv`. I assume that missing value means no sales occurs during that week. 

2. Shift: Shift is used to reticfy the error **weekly sales record itself bring about**. Here is brief explanation:
- X-mas season heat the pick sales record
- It fell on the week 52 from 2010 to 2012, given that weeks ends on Friday.
- However sales opportunities are different. it means:
	- 2010. 12. 25(Sat, 52th Week): 52 Week **didn't take advantage** of the x-mas season
	- 2011. 12. 25(Sun, 52th Week): 52 Week had only **1 day(Sat)** to take advantage of the x-mas season
	- test data) 2012. 12. 25(Tue, 52th Week): 52 Week had **3 day(Sat, Sun, Mon)** to take advantage of the x-mas season
- Thus, after predicting weekly sales, It needs to correct 52th week sales like by shift(exchange) 2.5 day with 51th week.

<img src='compare.png'>

## Limitation
- python3 module `auto_arima` much slower than `R`'s module. It took a couple days to make model by `auto_arima` and so It was hard to compare performances with others'.

