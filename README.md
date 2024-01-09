# Your Python Package README

## Installation

#### Via source
1. Clone [git repo](https://github.com/KDM-LAB/egises)

```
git clone git@github.com:KDM-LAB/egises.git
```

2. Install via pip (in your virtualenvironment)
```
pip install -e .
```

#### Via git repo
```
pip install git+ssh://git@github.com/KDM-LAB/egises.git
```
## Usage

```
import Egises, Document, Summary
# write document generator using Document, Summary class
def get_model_documents():
  # for document in docs:    
  #  yield Document(document)
  pass
# instantiate class
eg = Egises(model_name=model_name, measure=measure,
                    documents=utils.get_model_documents(model_name, CONSOLIDATED_FILEPATH),

# populate distances(resumable, starts from where left off)
eg.populate_distances()
# calculate scores
eg_score, accuracy_score = eg.get_egises_score(sample_percentage=sample_percentage)
```
### sample usage found [here](https://github.com/KDM-LAB/Evaluation-Framework-for-Personalized-Summarization/blob/main/EGISES_subjectivity/evaluation_script.py)
