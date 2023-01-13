# Codes for Multi-agent Bandit with Agent-Dependent Expected Rewards

***
This repository contains the codes for the simulations of C-CBGE UCBGE BGE and DDUCB


C-CBGE : confidence-weighted collective Boltzmann-Gumbel exploration.
         C-CBGE is an Boltzmann exploration based method for agent-dependent multi-agent bandit problems.

UCBGE : unweighted collective Boltzmann-Gumbel exploration.
        UCBGE is the agent-independent version of C-CBGE

BGE : Boltzmann-Gumbel exploration.
      BGE is a classic exploration policy for single-agent bandit problems

DDUCB : decentralized delayed upper confidence bound algorithm.
        DDUCB is a upper confidence bound based method for agent-indentpendent multi-agent bandit problems.




***
## Code structure

### C-CBGE:
codes for the algorithm called as confidence-weighted collective Boltzmann-Gumbel exploration (C-CBGE), C-CBGE is an Boltzmann exploration based method for agent-dependent multi-agent bandit problems.

env_CCBGE.py : contains code of the bandit problem environment for C-CBGE

agent_CCBGE.py : implementation of C-CBGE

data_rw.py : contains code for writing the data in an .xls file

learning_curve_CCBGE.py : run C-CBGE for the data in Fig.1 and Fig.2

er_constant_eta.py : run C-CBGE for the data in Fig.3

er_constant_N_full.py : run C-CBGE for the data in Fig.4

### DDUCB:
codes for the algorithm called as decentralized delayed upper confidence bound algorithm (DDUCB)

env_DDUCB.py : contains code of the bandit problem environment for DDUCB

agent_DDUCB.py : implementation of DDUCB

data_rw.py : contains code for writing the data in an .xls file

learning_curve_DDUCB.py : run DDUCB for the data in Fig.1 and Fig.2

### UCBGE & BGE:
codes for the algorithm called as unweighted collective Boltzmann-Gumbel exploration (UCBGE)

agent_UCBGE.py : implementation of UCBGE and BGE

env_UCBGE_BGE.py : contains code of the bandit problem environment for UCBGE and BGE

data_rw.py : contains code for writing the data in an .xls file

learning_curve_UCBGE_BGE.py : run DDUCB for the data in Fig.1 and Fig.2

### fig_plot:
contains the codes and data to plot Fig. 1, Fig. 2, Fig. 3 and Fig. 4

comparision_figS1.py : plot the subfigures in Fig.1

comparision_figS2.py : plot the subfigures in Fig.2

c_eta_fig.py : plot Fig. 3

c_number_fig.py : plot Fig. 4

