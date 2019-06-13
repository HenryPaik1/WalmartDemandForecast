import pandas as pd
from utils import *
from models import *

df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

df, test = init_df(df, test)
print('data loaded')
print('start fitting')

svd_stl_test = run_svd_stl_ets(df, test)
# svd_test = run_svd_ets(df, test)
# pca_stl_non_seasonalarima_test = run_pca_stl_non_seasonalarima(df, test)
# pca_custom = run_pca_custom_sarima(df, test)

# load prediction model and adjust datetime index to meet test data's 
filled_test = fill_test('test_test_svd_ets_ans.csv')

sub = pd.read_csv('sampleSubmission.csv')
sub.loc[:, 'Weekly_Sales'] = filled_test['Weekly_Sales'].values

# apply shift to submission
shifted_submission = shift_2_5(sub)
shifted_submission.to_csv('model_svd_ets_submission.csv', index=False)