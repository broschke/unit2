
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd

loansData = pd.read_csv('https://raw.githubusercontent.com/Thinkful-Ed/curric-data-001-data-sets/master/loans/loansData.csv')



loansData.dropna(inplace=True)

cleanFICORange = loansData['FICO.Range'].map(lambda x: x.split('-'))

cleanFICORange = cleanFICORange.map(lambda x: [int(n) for n in x])

loansData['FICO.Range'] = cleanFICORange

fico_score = [i[0] for i in loansData['FICO.Range']]

loansData['FICO.Score'] = fico_score
         
loansData['annual_inc'] = loansData['Monthly.Income'].map(lambda x: x * 12)
         
cleanInterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))
cleanLoanLength = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))

#print(loansData['Home.Ownership'])

#X = loansData['']

home_own = pd.Categorical(loansData['Home.Ownership']).codes

                 
intrate = cleanInterestRate
annual_inc = loansData['annual_inc']
loanamt = loansData['Amount.Requested']
loanlength = cleanLoanLength

y = np.matrix(intrate).transpose()
x1 = np.matrix(annual_inc).transpose()
x2 = np.matrix(loanamt).transpose()
x3 = home_own

x = np.column_stack([x1, x2, x3])

X = sm.add_constant(x)

model = sm.OLS(y,X)
f = model.fit()

print(f.summary())
