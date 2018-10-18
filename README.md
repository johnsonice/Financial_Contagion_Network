# Financial Contagion in a Multilayer Network
This is the code repository for the IMF working paper ["Financial Contagion in a Multilayer Network", by Yevgeniya Korniyenko, Manasa Patnam, Rita Maria Del Rio-Chanona, and Mason A. Porter](https://www.imf.org/en/Publications/WP/Issues/2018/05/15/Evolution-of-the-Global-Financial-Network-and-Contagion-A-New-Approach-45825)

### Setup 
```
git pull https://github.com/johnsonice/Financial_Contagion_Network.git
cd Financial_Contagion_Network
pip install -r requirements.txt
```

### Folder Structure
- ./data : raw input data and all adjacency matrix 
- ./lib : python utility modules being used in other scripts 
- ./results : network structural measurements; contagion results; charts etc
- ./src : materals for readme 
- 01* - 06* : scripts to reproduce all results and charts in the paper 


### Results
![alt text](./src/figure1.PNG)
![alt text](./src/figure4.PNG)
![alt text](./src/figure5.PNG)
![alt text](./src/figure6.PNG)