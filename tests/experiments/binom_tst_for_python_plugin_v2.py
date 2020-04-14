from scipy.stats import binom_test
result=df
result["p_value"] = result.apply(lambda row : binom_test_v2(row.Events_B,row.Events_A + row.Events_B,row.Uptime_B/(row.Uptime_A +row.Uptime_B),alternative='greater'), axis=1)
result["p_value_two_sided"] = result.apply(lambda row : binom_test_v2(row.Events_B,row.Events_A + row.Events_B,row.Uptime_B/(row.Uptime_A +row.Uptime_B)), axis=1 )
