### Instructions

Getting started 

First create a virtual environment:

```console
python -m venv .venv
source ./.venv/bin/activate
```
For windows do 
```console
source ./.venv/scripts/activate
```
Install packages (only ones for this one are torch and matplotlib)
```console
pip install -r requirements.txt
```
## Training process for fixed-masks MCMC
 * The main model, Net and NetV2, is in model.py.
 * Each model consistents of a ConsistentMCDropout layer which allows for the use of fixed masks adopted from here: https://blackhc.github.io/batchbald_redux/consistent_mc_dropout.html 
 * Here is an example of how to create such a layer:
```python
dropout1 = ConsistentMCDropoutMask(p=0.8)
```
* This layer functions by keeping a dictionary of bit-maps where a given bit map is invoked based on the map that is called
* During training masks were partitioned into nGroups where nGroups < nMasks to ensure that different masks had enough training data to work with

## MCMC 
* The proposal distribution used for MCMC was a non-adaptive uniform distribution (each mask had equal probability
* Create with the following line:
```python
mcmc = MCMC(increment_amt = 1)
mcmc.transition(x, y) # Several times to change distribution
```
Where increment_amt functions as a way to weight sucesses more. A higher value allows for resulting distributions to be more skewed or favored towards certain masks
* In order to perform inference we do
 ```python
predict(x, chosen_masks)
```
Where chosen_masks is a tensor which contains the masks that are to be used. In this case, these masks will 'absorb' the probability from the unused masks to make up the distribution
* On inference, we can use the weighted average of which masks that are accepted instead sampling from accepted samples; this is because we are dealing with a smaller and discrete distirbution
  
## train.py arguements:

| Argument            | Short Flag | Type    | Default Value   | Required | Description                                      |
|---------------------|-------------|---------|-----------------|-----------|--------------------------------------------------|
| `--epochs`           | `-e`         | `int`   | `12`             | No        | Number of training epochs                         |
| `--batch_size`       | `-b`         | `int`   | `32`             | No        | Batch size for training                           |
| `--lr`               | `-l`         | `float` | `0.001`          | No        | Learning rate                                      |
| `--num_masks`        | `-nm`        | `int`   | N/A              | **Yes**   | Number of masks to train on                         |
| `--dropout_probs`    | `-dp`        | `float` | N/A              | **Yes**   | List of dropout probabilities (need 2)                      |
| `--num_groups`       | `-ng`        | `int`   | N/A              | **Yes**   | Number of partitions in training data. Each mask trains on a single partition        |
| `--increment_amt`      | `-i`         | `int`   | `1`              | No        | Amount to increment a distribution by when accepted                              |
| `--path`             | `-p`         | `str`   | `"experiments"`  | No        | Path to place experiment in                         |




# Results
See mask_training.ipynb for the training of the fixed-mask method.


### Shallow network results:
![image](https://github.com/user-attachments/assets/3c97f293-190f-4eca-8d1e-42badb2f34ca)

### Deep network results: 
![image](https://github.com/user-attachments/assets/2a9459af-0d8e-4596-b956-22011ea4d54d)




