## Description

In this directory we implement a set of tools for inspecting various methods WSI and LSCD tasks.

Currently, it is a web application based on Python/Flask.

We deploy it as a set of docker containers for easier dependency management.

#### Dependencies

The web app depends on some code outside of its directory, namely on `../substwsi/max_ari.py` and `../substwsi/subts_loading.py`.
So to run the web app it needs to be invoked from the repository root (summer-wsi) directory, which is parent to the app's root directory.
To make sure imports from the `../substwsi` directory work, make sure that all imports within the `substwsi` folder have absolute paths.

Incorrect: `from wsi import clusterize_search, get_distances_hist`

Correct: `from substwsi.wsi import clusterize_search, get_distances_hist`

Files to fix relative imports are `wsi.py` and `max_ari.py` in `substwsi` folder.

You may also encounter errors in `../substwsi/wsi.py`, in particular, incorrect import for `import sklearn.cluster.hierarchical`. This is related to Sci-Kit library version.
Unfortunately there is no quick fix for that just yet, but you can safely remove that import and all of the usage of `sklearn.cluster.hierarchical` from `wsi.py`.

#### Running the web app

To run the tool, you need to ensure you have [docker](https://www.docker.com/) installed on your machine.

Navigate to `summer-wsi` directory and run the following command in the terminal.

```
PORT=[PORT] docker-compose up
```
Where port is the port on which you want to run the web server. 
If you are not sure what this means simply put any number between 8000 and 9000.
You may encounter an error if the specified port turns out to be occupied on your machine, you can simply try another one.

To launch the web app navigate to `localhost:[PORT]` 

#### Reproducing experiments

1. download the datasets for three target words at https://tinyurl.com/MethodsAndToolsForLSCD 
2. Use the upload dataset page to add datasets to the system 
3. use `<mask><mask>-(а-также-T)-2ltr2f0s_topk150_fixspacesTrue.npz+0.0+` for input file names fields
4. give any descriptive name to the dataset in the dataset id field
5. give descriptive names (e.g. pre-Soviet, post-Soviet) to corpus names
6. select the pre_*.zip files downloaded on step 1 for first corpora file input, and post_*.zip for second
7. click upload and navigate to home page
8. (you may need to wait until the upload and dataset processing finishes, it may take several minutes or even more)
9. when on the home page, on the left panel the dataset name appears (you'll need to refresh the page) in the drop-down list `choose dataset` that means it is ready
10. choose the dataset in the list, set the parameters (check next section for set of parameters to use) and click `Run experiment` at the top of the left panel
11. the WSI task will start, it may take several minutes, even dozens depending on your computer and the size of the dataset
12. Once the processing is finished, an entry on the home page will appear with the time when the experiment started and finished, and with the dataset name
13. Click the load button and refer to the thesis section 3.2.4 as a user's tutorial

##### Random seeds for exact experiment reproduction

To reproduce exact results of experiments please use the following set of parameters

For all words
```
Clusterer
   AgglomerativeClusterer
      n_clusters: [2, 3, 4, 5, 6]
      linkage: ['average']
      affinity: ['cosine']

Vectorizers
   SubstsTfidfVectorizer
      analyzer: ['word']
      min_df: [0.03]
      max_df: [0.8]
      topk: [150]
```

Word машина

```
Number of samples: 2500
Random seed: 2
```

Word пакет

```
Number of samples: 1000
Random seed: 1
```

#### Shutting down the web app

Navigate to `summer-wsi` directory and run the following command in the terminal.

```
PORT=[PORT] docker-compose down
```