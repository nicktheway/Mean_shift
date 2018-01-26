# Mean_shift
A cuda implemented mean-shift algorithm

## To compile the programs:
```
> cd source
> make
> make seed
```
## To run the programs:
#### Shared memory implementation with the default (r15) data
`> ./shms` 

#### Global memory implementation with the default (r15) data
`> ./glms`

#### Shared memory implementation with the SEED_DATASET: [seed data](https://archive.ics.uci.edu/ml/datasets/seeds)
`> ./sshms` 

#### Global memory implementation with the SEED_DATASET: [seed data](https://archive.ics.uci.edu/ml/datasets/seeds)
`> ./sglms`

To clear the `.o` files:
`make clean`
