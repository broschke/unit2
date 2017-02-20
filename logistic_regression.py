import statsmodels.api as sm
import pandas as pd
import math

df = pd.read_csv('C:\\Users\\Bernardo.Roschke\\Dropbox\\Thinkful\\data\\unit 2\\loansData_clean.csv')

df.dropna(inplace=True)

df['IR_TF'] = [1 if x >= .12 else 0 for x in df['Interest.Rate']]

ir_tf = df['IR_TF']

df.drop(['IR_TF'],inplace=True,axis=1)

df['Intercept'] = 1
  
ind_vars = ['Amount_Requested', 'FICO_Score', 'Intercept']

logit = sm.Logit(ir_tf.astype(float), df[ind_vars].astype(float))

result = logit.fit()

coeff = result.params
print(coeff)

#p(x) = 1/(1 + e^(intercept + 0.087423(FicoScore) − 0.000174(LoanAmount))

prob = 1 / (1 + math.exp(result.params.Intercept + result.params.FICO_Score*(720) - result.params.Amount_Requested*(10000)))

print(prob)