from scipy.stats import binom_test
result=df
p_vals = []
for i in range(df.shape[0]):
  n1 = float(df.iloc[[i]]["n"])
  t1 = float(df.iloc[[i]]["t"])
  n2 = 8234
  t2 = 8948595990.85353
  p_val = binom_test(n1,n1+n2,t1/(t1+t2),alternative='greater')
  p_vals.append(p_val)
result["p_value"]=np.array(p_vals)

