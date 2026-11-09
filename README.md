# Real-Time Systems – Lab 4 (Unit 4 Mini Projects)

University of Rwanda, College of Science and Technology, School of ICT, Department of Computer Engineering
Group 3: Ronsard RUGIRANGOGA (223026673), Ivan MULISA (223026695), Dieudonne BYIRINGIRO (223026513)

The written report summarises the results; this repository holds the complete code, the printed results and all charts.

## Structure

```
Mini_Project_4_1_Sampling_Jitter/
    rt_sampling_sim.py   fixed-period scan loop with jitter and overruns, threshold and P controllers, sweep,
                         compliance > 1.5 T, halved period -> results_4_1.txt, figures/ (fig1_*, fig2_* per configuration)
Mini_Project_4_2_MultiRate_Pipeline/
    pipeline.py          3 or 4 sensor streams, FP / EDF / round-robin scheduling, on-time / late / missed report,
                         threshold alerts with latency -> results_4_2.txt, figures/ (signals_* per configuration, compliance_bars)
Mini_Project_4_3_Edge_Cloud/
    edge_cloud_sim.py    edge path, long-tailed cloud path, shared event stream, hybrid policies, congestion sweep
                         -> results_4_3.txt, figures/
```

## Run

Python 3.10 or later with matplotlib (`pip install -r requirements.txt`); everything else is the standard library.

```
python Mini_Project_4_1_Sampling_Jitter/rt_sampling_sim.py       (~5 s)
python Mini_Project_4_2_MultiRate_Pipeline/pipeline.py           (~3 s)
python Mini_Project_4_3_Edge_Cloud/edge_cloud_sim.py             (~3 s)
```

Seeds are fixed, so the numbers match the report.
