# Walmart Demand Forecasting
## time series analysis
- Kaggle: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data
- reference: 
	- kaggle winner's code
	- github: https://github.com/davidthaler/Walmart_competition_code
	- explanation: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8125#latest-357454 
	- key adjustment: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8028
- Models
	- PCA + ETS(ExponenTial Smoothing): score 2609.84528(best)
	- PCA + stl + SARIMA: score 2689.39573
	- PCA + customed seasonal adjust+ SARIMA: for seasonal adjustment,  I dedcut the average of every week which is considered deterministic seasonality (eg. avg(2010/2011/2012 1st week), ..., avg(2010/2011 52th week))

## Key concept

1. PCA: PCA is used to find signal through the specific department. It additionally make it easier to deal with missing values which is NA or which is not exist in `train.csv`, but in `test.csv`. I assume that missing value means no sales occurs during that week. 

2. Shift: Shift is used to reticfy the error **weekly sales record itself bring about**. Here is brief explanation:
- X-mas season heat the pick sales record
- It fell on the week 52 from 2010 to 2012, given that weeks ends on Friday.
- However sales opportunities are different. it means:
	- 2010-12-25(Sat, 52nd Week): 52nd Week **didn't take advantage** of the x-mas season
	- 2011-12-25(Sun, 52nd Week): 52nd Week had only **1 day(Sat)** to take advantage of the x-mas season
	- test data) 2012-12-25(Tue, 52nd Week): 52nd Week had **3 day(Sat, Sun, Mon)** to take advantage of the x-mas season
- Thus, after predicting weekly sales, It needs to correct 52nd week sales like by shift(exchange) 2.5 days with 51st week.

<img src='compare.png'>

## Limitation
- python3 module `auto_arima` much slower than `R`'s module. It took a couple days to make model by `auto_arima` and so It was hard to compare performances with others'.

