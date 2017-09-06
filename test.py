import template as t
import pandas as pd
import numpy as np

phi=t.get_feature_matrix('data/train.csv')
print('DONE GENERATING FEATURES')
Y=t.get_output('data/train.csv')
print('DONE GENERATING OUTPUT')
wts=t.get_weight_vector(phi,Y,0.0,2)
print('DONE GENERATING WEIGHTS')
wts.dump('wts.pickl')
test_phi=t.get_feature_matrix('data/test.csv')
print('DONE GENERATING TEST FEATURES')
test=pd.read_csv('data/test.csv')
result=pd.DataFrame(test['Id'])
result['Output']=pd.Series(np.dot(test_phi,wts).flatten())
result.to_csv('output.csv',index=False)
print('DONE GENERATING OUTPUT')
