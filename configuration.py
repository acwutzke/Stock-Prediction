

#### Training.py parameters ####

ticker_list_file='ticker_lists/US_METALS.csv'
ticker_exchange='' # leave blank unless downloading TSX stocks ('.TO')
ticker_sample_size=30
ticker_col_name='SYM'


training_data_start='2015-01-01'
training_data_end='2019-12-31'

test_data_start='2019-10-01'
test_data_end='2021-03-20'

# choose whether to save the model - will be saved in models folder
save_model=True
model_name='US_METALS_model'

# to change change index with which to generate features and evaluate performance
index='SPY'

#### eval.py parameters ####

# name of model you would like to evaluate
eval_model_name=model_name

show_auc_roc_curve=False
show_prec_recall=False

# backtest inputs
show_backtest=True
backtest_conf=.5
backtest_max_positions=10
backtest_cash=100000
backtest_index='SPY'
backtest_title='Backtest Results'
