# Stage-B actual post-action interference comparison

Paired seeds: 41; warm-up excluded: 1000 TTI.

| metric | PFQ mean | AI mean | paired AI - PFQ |
|---|---:|---:|---:|
| TTI mean INR | 418.883407 | 372.880667 | -46.002740 |
| TTI mean interference (dB over noise) | 26.220932 | 25.715699 | -0.505233 |
| Pooled scheduled-link mean INR | 421.528904 | 376.078560 | -45.450344 |
| Pooled interfered-link mean INR | 423.200630 | 384.050562 | -39.150068 |
| Interfered scheduled-link ratio | 99.604980% | 97.924231% | -1.680749% |
| Goodput (Mbps) | 1539.093376 | 1542.189781 | 3.096405 |

INR is computed after the scheduling action from cross-cell channel matrices, the actually selected interferer UE precoder, PRG occupancy, and selected layer count.
