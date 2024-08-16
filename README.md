# model configs:

# Regression:
### For deep ensemble
#### 100 samples, 50 models, 400 epochs: 153 (First Case)
#### 100 samples, 10 models, 400 epochs: 32 (Second Case)
#### 100 samples, 5 models, 400 epochs: 13 (Third Case)

# Equivalents (equivalents must use same number of training samples, the rest can be different):
### First Case

#### Bootstrap 100, 50, 1000 (165)
#### MC Dropout 100, 5/10/50, 1000 (7) is the longest it takes to train
#### LA 100, 100/250/1000, 1000 (5) is the longest it takes to train
#### HMC 100, 1000, 1000 (212)
#### VI 100, 1000, 1000 (22) is the longest it takes to train

### Second Case
#### same for all except for HMC and Bootstrap
#### HMC 100, 100, 400 (32)
#### Bootstrap 100, 50, 200 (37)

### Third Case
#### same for all except for HMC and Bootstrap
#### HMC 100, 100, 1000 (24)
#### Bootstrap 100, 5, 400 (16)


### For deep ensemble
#### 30 samples, 50 models, 400 epochs: 36 (First Case)
#### 30 samples, 10 models, 400 epochs: 7 (Second Case)
#### 30 samples, 5 models, 400 epochs: 4 (Third Case)

# Equivalents (equivalents must use same number of training samples, the rest can be different):
### First Case

#### Bootstrap 30, 50, 400 (46)
#### MC Dropout 30, 5/10/50, 1000 (3) is the longest it takes to train
#### LA 30, 100/250/1000, 1000 (2) is the longest it takes to train
#### HMC 30, 100, 400 (27)

### Second Case
#### same for all except for HMC and Bootstrap
#### HMC 30, 100, 200 (24)
#### Bootstrap 30, 10, 400 (7)

### Third Case
#### same for all except for HMC and Bootstrap
#### HMC 30, 100, 200 (24)
#### Bootstrap 30, 5, 400 (4)