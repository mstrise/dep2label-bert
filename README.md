# Dependency Parsing as Sequence Labeling with BERT

This is the source code for the paper "A Unifying Theory of Transition-based and Sequence Labeling Parsing" accepted at COLING2020:

### Overview of the Encoding Family for Dependency Parsing in SL

| Name       | Type of encoding         | supports non-projectivity?  | 
| ------------- |:-------------:| :-------------:|
| ```rel-pos```     | Relative Part-of-Speech-based | :heavy_check_mark: |
| ```1-planar-brackets```     | Bracketing-based      |  :heavy_check_mark: / &#10007; | 
| ```2-planar-brackets-greedy``` |   Second-Plane-Averse Greedy Plane Assignment    |   :heavy_check_mark:  | 
| ```2-planar-brackets-propagation```  |   Second-Plane-Averse Plane Assignment based on Restriction Propagation on the Crossings Graph   |   :heavy_check_mark:  |
| ```arc-standard```  |  Arc Standard in Transition-based   |   &#10007;  | 
| ```arc-eager```  |  Arc Eager in Transition-based   |   &#10007;  | 
| ```arc-hybrid```   | Arc Hybrid in Transition-based    |   &#10007;  | 
| ```covington```  |  Covington in Transition-based    |   :heavy_check_mark:  | 

## Requirements

We have modified the script from [huggingface🤗](https://huggingface.co/) in order to enable training of dependency parsing as sequence labeling with BERT. The code is based on the repository for [Discontinuous Constituent Parsing as Sequence Labeling](https://github.com/aghie/disco2labels).

It is recommended to create a virtual environment in order to keep the installed packages separate to avoid conflicts with other programs.

```sh
pip install -r requirements.txt
```

## Scripts for encoding and decoding dependency labels


To encode a CoNNL-X file to SL file:

```bash
python encode_dep2labels.py --input --output --encoding [--mtl] 
```
where
```bash
input=...    # CoNNL-X file to encode 
output=...   # output file with encoded dependency trees as labels (OBS with extension ".tsv")
encoding=... # encoding type= ["rel-pos", "1-planar-brackets", "2-planar-brackets-greedy","2-planar-brackets-propagation","arc-standard", "arc-eager","arc-hybrid", "covington","zero"]
mtl=...      # optionally, nb of multi-tasks= ["1-task","2-task","2-task-combined","3-task"]. By default, type that gives the best results is chosen
```

To decode a SL file to a CoNNL-X file:
```bash
python decode_labels2dep.py --input [--conllu_f] --output --encoding 
```
where
```bash
input=...    # SL file to decode 
conllu_f=... # optionally, the corresponding CoNNL-X file (in case of special indexing i.e. 1.1 or 1-2)
output=...   # output file with decoded dependency trees to CoNNL-X format
encoding=... # encoding type= ["rel-pos", "1-planar-brackets", "2-planar-brackets-greedy","2-planar-brackets-propagation","arc-standard", "arc-eager","arc-hybrid", "covington"]
```

## Training a model

Modify and run the following script:

```bash
./train.sh
```


## Parsing with a trained model

Modify and run the following script:

```bash
./parse.sh
```

## Acknowledgements

This work has received funding from the European Research Council (ERC), under the European Union's Horizon 2020 research and innovation programme (FASTPARSE, grant agreement No 714150).

## Reference

If you wish to use our work for research purposes, please cite us!

```
@inproceedings{strzyz-etal-2019-viable,
    title = "Viable Dependency Parsing as Sequence Labeling",
    author = "Strzyz, Michalina  and
      Vilares, David  and
      G{\'o}mez-Rodr{\'\i}guez, Carlos",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1077",
    doi = "10.18653/v1/N19-1077",
    pages = "717--723",
    abstract = "We recast dependency parsing as a sequence labeling problem, exploring several encodings of dependency trees as labels. While dependency parsing by means of sequence labeling had been attempted in existing work, results suggested that the technique was impractical. We show instead that with a conventional BILSTM-based model it is possible to obtain fast and accurate parsers. These parsers are conceptually simple, not needing traditional parsing algorithms or auxiliary structures. However, experiments on the PTB and a sample of UD treebanks show that they provide a good speed-accuracy tradeoff, with results competitive with more complex approaches.",
}
```

