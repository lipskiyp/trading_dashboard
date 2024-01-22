# Pandas Trading Dashbaord

Pandas practice project that calculates metrics for financial time-series.

## Example Output

```bash
                            benchmark               ts_1              ts_2
Avg. Price Change             0.06783          -0.062331         -0.092221
Avg. Annual Return           0.190378          -0.144275         -0.209066
CAGR                          0.20279          -0.138661          -0.19213
Var                          0.009869           0.011089          0.010094
Vol                          0.099343           0.105305          0.100471
DownsideVol                  0.176494           0.196947          0.198266
UpsideVol                    0.011422             0.0177          0.013514
CoVar                        0.009869          -0.000021          0.000511
Corr                              1.0          -0.001972          0.051221
Skew                              0.0            0.00001          0.000009
CoSkew                            0.0          -0.000003          0.000004
Kurt                         0.000145           0.000228          0.000189
CoKurt                       0.000145          -0.000006          0.000016
DD                          -0.183349          -0.009844         -0.010982
DDur                355 days 00:00:00    7 days 00:00:00  63 days 00:00:00
MaxDD                       -0.183349          -0.076455         -0.085263
MaxDDur             355 days 00:00:00  101 days 00:00:00  59 days 00:00:00
PainIdx                       0.10961           0.018286          0.020202
```

### /dashborad

Contains trading dashboard class..

### /timeseries

Contains random time-series factory.
