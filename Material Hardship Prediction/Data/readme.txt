
The Fragile Families Challenge

Version: 16 January 2018

This zipped folder contains six comma-separated values files, a text codebook file, and a Stata .dta file 
that contains the data with variable and value labels.

The following files were available to participants in the Fragile Families Challenge:
- background.csv
- background.dta
- train.csv
- prediction.csv
- codebook_FFChallenge.txt


The following files were only available to organizers:
- leaderboard.csv
- leaderboardUnfilled.csv
- test.csv

************

background.csv contains 4,242 rows (one per child) and 12,943 columns

background.dta contains the same information, plus variable and value labels, in a Stata data file.

These files contain:

challengeID: A unique numeric identifier for each child.


12,942 background variables asked from birth to age 9, which you may use in building your model.

Full documentation is at http://www.fragilefamilies.princeton.edu/

An intro to the documentation is at http://www.fragilefamilieschallenge.org/survey-documentation/

************

train.csv contains 2,121 rows (one per child in the training set) and 7 columns.

These are the outcome variables measured at approximately child age 15, which you can use to train your models.

The file contains:

challengeID: A unique numeric identifier for each child.

Six outcome variables (each variable name links to a blog post about that variable)

Continuous variables: grit, gpa, materialHardship

Binary variables: eviction, layoff, jobTraining

************

prediction.csv contains 4,242 rows and 7 columns.

This file is provided as a skeleton for your submission; you will submit a file in exactly this form but with 
your predictions for all 4,242 children included.

The file contains:

challengeID: A unique numeric identifier for each child.

Six outcome variables, as in train.csv. These are all filled with the mean value in the training set.

************

leaderboard.csv contains 4,242 rows and 7 columns.

*This file was only available to organizers, not participants.*

This file contains the outcome values for the 1/8 of observations used to construct the leaderboard set. 
During the Challenge, the leaderboard set provided respondents with instant feedback on their submissions through the Codalab web platform.

For the 7/8 of rows not included in the leaderboard set, all outcomes are marked NA. 
For the 1/8 of rows in the leaderboard set, missing values are replaced with random draws from the observed values. 
This was the file used to score submissions to the Codalab platform. On Codalab, NA values were ignored in scoring. 
See more information at http://www.fragilefamilieschallenge.org/understanding-your-score-on-the-holdout-set/.

The file contains:

challengeID: A unique numeric identifier for each child.

Six outcome variables, as in train.csv. These are all filled with the observed outcomes when available, 
or a random draw from observed outcome in place of a missing value. Rows representing observations not in the leaderboard set are marked NA.

************

leaderboardUnfilled.csv contains 4,242 rows and 7 columns.

*This file was only available to organizers, not participants.*

This file contains the outcome values for the 1/8 of observations used to construct the leaderboard set. During the Challenge, 
the leaderboard set provided respondents with instant feedback on their submissions through the Codalab web platform.

For the 7/8 of rows not included in the leaderboard set, all outcomes are marked NA. For the 1/8 of rows in the leaderboard set, 
missing values are also marked NA. The version used on Codalab contained randomly imputed values for missing cases; see leaderboard.csv.

The file contains:

challengeID: A unique numeric identifier for each child.

Six outcome variables, as in train.csv. These are all filled with the observed outcomes when available for rows representing observations in the leaderboard set. 
Rows representing observations not in the leaderboard set and missing observations are marked NA.

************

test.csv contains 1,591 rows and 7 columns.

*This file was only available to organizers, not participants.*

These are the outcome variables measured at approximately child age 15, which you can use to evaluate your models.

This file contains the outcome values for the 3/8 of observations that were held out and used to evaluate all submissions at the end of the Challenge. 
Missing values are marked NA and were ignored in scoring.

The file contains:

challengeID: A unique numeric identifier for each child.

Six outcome variables, as in train.csv. These are all filled with the observed outcomes when available and NA in the case of missing values.

************

codebook_FFChallenge.txt is a text file that contains the codebook for all variables in the Challenge data file. 
This combines several codebooks from the main Fragile Families and Child Wellbeing Study documentation.

You should also refer to the full documentation at http://www.fragilefamilies.princeton.edu/
for questionnaires and more detailed information.