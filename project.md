## My Project

Predicting Southern Great Plains Afternoon Precipitation Events

***

## Introduction 

Here is a summary description of the topic. Here is the problem. This is why the problem is important.

- Rainfall forecasting important for these reasons: extreme weather events (e.g., hurricanes, rain- and snow-storms) often create dire situations (e.g., floods, landslides, and wildfires) causing severe economic losses and casualties. As global warming continues, the
frequency and intensity of extreme weather events are likely to increase in many regions (https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2022GL097904) 
  
- Machine learning approach important for these reasons: existing models use complex statistical models that are often too costly, both computationally and budgetary, or are not applied to downstream applications (https://www.sciencedirect.com/science/article/pii/S266682702100102X)
  
- SGP hot spot for land-atmosphere interactions: Numerous studies have addressed the effect of soil moisture on
subsequent rainfall, yet consensus remains elusive (https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2018GL078598) 

There is some dataset that we can use to help solve this problem. This allows a machine learning approach. This is how I will solve the problem using supervised/unsupervised/reinforcement/etc. machine learning.

- 

We did this to solve the problem. We concluded that...

## Data

Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!

![](assets/IMG/datapenguin.png){: width="500" }

*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

## Modelling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] DALL-E 3

[back](./)

