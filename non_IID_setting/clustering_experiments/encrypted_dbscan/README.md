# Federated DBSCAN under Encryption

The DBSCAN-based FL-HPO can also be performed under encryption by adjusting the [federated dbscan code](https://github.com/GM862001/F_DBSCAN) and using the [lattigo](https://github.com/tuneinsight/lattigo) library.

## Implementation
The implementation is split into several directories:

### `comparator`

**Description**: Directory contains all functions needed for performing comparisons under encryption. The basic implementation was taken from the **Heavy-Hitters** project and extended with collective bootstrapping.


### `dbscan`

**Description**: Directory contains the core part of the federated DBSCAN algorithm and defines the basic objects: Point, Cell, Client. 

**Files**:

- `cell`: defines the Cell object and how to retrieve adjacent cells for a cell
- `client`: defines all functions run locally on a Client: local grid creation, leveraging dense mask to aggregate local points into clusters, outlier removal, key pair generation, saving best parameters, unscaling lr/mom params, etc.
- `constants`: defines DBSCAN constants, as well as constants used for scaling and parametrizing encrypton functions.
- `point`: defines the Point object and how to retrieve the closest adjacent cell for a point
- `reader`: reads the experiments and creates clients/parties
- `utils`: defines helper functions for printing, sorting, shuffling arrays/vectors.

### `encrypted`
**Description**: Directory contains the encrypted part of the federated DBSCAN algorithm: grid aggregation, threshold comparison and parameter averaging. 

**Files**:

- `averaging`: acc/lr/mom averaging under encryption
- `gridAggregation`: grid aggregation and encrypted comparison with a threshold (i.e., dense mask calculation), implements collective bootstrapping
