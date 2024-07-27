#!/usr/bin/env python
# coding: utf-8

# # **TikTok Project**
# **Course 6 - The nuts and bolts of machine learning**

# Recall that you are a data professional at TikTok. Your
# supervisor was impressed with the work you have done
# and has requested that you build a machine learning model that can be used to determine whether a video contains a claim or whether it offers an opinion. With a successful prediction model, TikTok can reduce the backlog of user reports and prioritize them more efficiently.
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions.

# # **Course 6 End-of-course project: Classifying videos using machine learning**
# 
# In this activity, you will practice using machine learning techniques to predict on a binary outcome variable.
# <br/>
# 
# **The purpose** of this model is to mitigate misinformation in videos on the TikTok platform.
# 
# **The goal** of this model is to predict whether a TikTok video presents a "claim" or presents an "opinion".
# <br/>
# 
# *This activity has three parts:*
# 
# **Part 1:** Ethical considerations
# * Consider the ethical implications of the request
# 
# * Should the objective of the model be adjusted?
# 
# **Part 2:** Feature engineering
# 
# * Perform feature selection, extraction, and transformation to prepare the data for modeling
# 
# **Part 3:** Modeling
# 
# * Build the models, evaluate them, and advise on next steps
# 
# Follow the instructions and answer the questions below to complete the activity. Then, you will complete an Executive Summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.
# 
# 

# # **Classify videos using machine learning**

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## **Pace: Plan**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.
# 
# In this stage, consider the following questions:
# 
# 1.   **What are you being asked to do? What metric should I use to evaluate success of my business/organizational objective?**
# 
# 2.   **What are the ethical implications of the model? What are the consequences of your model making errors?**
#   *   What is the likely effect of the model when it predicts a false negative (i.e., when the model says a video does not contain a claim and it actually does)?
# 
#   *   What is the likely effect of the model when it predicts a false positive (i.e., when the model says a video does contain a claim and it actually does not)?
# 
# 3.   **How would you proceed?**

# **Responses:**

# **1. What are you being asked to do?**
# 
# **Business need and modeling objective**
# 
# TikTok users can report videos that they believe violate the platform's terms of service. Because there are millions of TikTok videos created and viewed every day, this means that many videos get reported&mdash;too many to be individually reviewed by a human moderator.
# 
# Analysis indicates that when authors do violate the terms of service, they're much more likely to be presenting a claim than an opinion. Therefore, it is useful to be able to determine which videos make claims and which videos are opinions.
# 
# TikTok wants to build a machine learning model to help identify claims and opinions. Videos that are labeled opinions will be less likely to go on to be reviewed by a human moderator. Videos that are labeled as claims will be further sorted by a downstream process to determine whether they should get prioritized for review. For example, perhaps videos that are classified as claims would then be ranked by how many times they were reported, then the top x% would be reviewed by a human each day.
# 
# A machine learning model would greatly assist in the effort to present human moderators with videos that are most likely to be in violation of TikTok's terms of service.

# **Modeling design and target variable**
# 
# The data dictionary shows that there is a column called `claim_status`. This is a binary value that indicates whether a video is a claim or an opinion. This will be the target variable. In other words, for each video, the model should predict whether the video is a claim or an opinion.
# 
# This is a classification task because the model is predicting a binary class.

# **Select an evaluation metric**
# 
# To determine which evaluation metric might be best, consider how the model might be wrong. There are two possibilities for bad predictions:
# 
#   - **False positives:** When the model predicts a video is a claim when in fact it is an opinion
#   - **False negatives:** When the model predicts a video is an opinion when in fact it is a claim
# 
# 
# 
# **2. What are the ethical implications of building the model?**
# In the given scenario, it's better for the model to predict false positives when it makes a mistake, and worse for it to predict false negatives. It's very important to identify videos that break the terms of service, even if that means some opinion videos are misclassified as claims. The worst case for an opinion misclassified as a claim is that the video goes to human review. The worst case for a claim that's misclassified as an opinion is that the video does not get reviewed _and_ it violates the terms of service. A video that violates the terms of service would be considered posted from a "banned" author, as referenced in the data dictionary.
# 
# Because it's more important to minimize false negatives, the model evaluation metric will be **recall**.
# 

# **3. How would you proceed?**
# 
# **Modeling workflow and model selection process**
# 
# Previous work with this data has revealed that there are ~20,000 videos in the sample. This is sufficient to conduct a rigorous model validation workflow, broken into the following steps:
# 
# 1. Split the data into train/validation/test sets (60/20/20)
# 2. Fit models and tune hyperparameters on the training set
# 3. Perform final model selection on the validation set
# 4. Assess the champion model's performance on the test set
# 
# ![](https://raw.githubusercontent.com/adacert/tiktok/main/optimal_model_flow_numbered.svg)
# 
# 
# 

# ### **Task 1. Imports and data loading**
# 
# Start by importing packages needed to build machine learning models to achieve the goal of this project.

# In[1]:


# Import packages for data manipulation
import pandas as pd
import numpy as np

# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import packages for data preprocessing
from sklearn.feature_extraction.text import CountVectorizer

# Import packages for data modeling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, \
recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance


# Load the data from the provided csv file into a dataframe.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

# ### **Task 2: Examine data, summary info, and descriptive stats**

# Inspect the first five rows of the dataframe.

# In[5]:


# Display first few rows
data.head()


# Get the number of rows and columns in the dataset.

# In[6]:


# Get number of rows and columns
data.shape


# Get basic information about the dataset.

# In[7]:


# Get basic information
data.info()


# Generate basic descriptive statistics about the dataset.

# In[8]:


# Generate basic descriptive stats
data.describe()


# Check for and handle missing values

# In[9]:


# Check for missing values
data.isna().sum()


# There are very few missing values relative to the number of samples in the dataset. Therefore, observations with missing values can be dropped.

# In[3]:


# Drop rows with missing values
data = data.dropna(axis=0)


# Check for and handle duplicates

# In[11]:


# Check for duplicates
data.duplicated().sum()


# There are no duplicate observations in the data.

# Check class balance.

# In[4]:


# Check class balance
data["claim_status"].value_counts(normalize=True)


# Approximately 50.3% of the dataset represents claims and 49.7% represents opinions, so the outcome variable is balanced.

# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## **PACE: Construct**
# Consider the questions in your PACE Strategy Document to reflect on the Construct stage.

# ### **Task 3. Feature engineering**
# 
# 
# Extract the length (character count) of each `video_transcription_text` and add this to the dataframe as a new column called `text_length` so that it can be used as a feature in the model.

# In[13]:


# Create `text_length` column
data['text_length'] = data['video_transcription_text'].str.len()
data.head()


# Calculate the average `text_length` for claims and opinions.
# 
# 

# In[14]:


data[['claim_status', 'text_length']].groupby('claim_status').mean()
# the code can also be written like this just that it won't appear as a table
#data.groupby('claim_status')['text_length'].mean()


# Visualize the distribution of `text_length` for claims and opinions using a histogram.

# In[15]:


# Visualize the distribution of `text_length` for claims and opinions
# Create two histograms in one plot

sns.histplot(data=data, stat="count", multiple="dodge", x="text_length",
             kde=False, palette="pastel", hue="claim_status",
             element="bars", legend=True)
plt.xlabel("video_transcription_text length (number of characters)")
plt.ylabel("Count")
plt.title("Distribution of video_transcription_text length for claims and opinions")
plt.show()


# Letter count distributions for both claims and opinions are approximately normal with a slight right skew. Claim videos tend to have more characters&mdash;about 13 more on average, as indicated in a previous cell.

# **Feature selection and transformation**

# Encode target and catgorical variables.

# In[16]:


X = data.copy()
# Drop unnecessary columns
X = X.drop(['#', 'video_id'], axis=1)
# Encode target variable
X['claim_status'] = X['claim_status'].replace({'opinion': 0, 'claim': 1})
# Dummy encode remaining categorical values
X = pd.get_dummies(X,
                   columns=['verified_status', 'author_ban_status'],
                   drop_first=True)
X.head()


# ### **Task 4. Split the data**

# Assign target variable.
# 
# In this case, the target variable is `claim_status`.
# * 0 represents an opinion
# * 1 represents a claim

# In[17]:


# Isolate target variable
y = X['claim_status']


# Isolate the features.

# In[18]:


# Isolate features
X = X.drop(['claim_status'], axis=1)

# Display first few rows of features dataframe
X.head()


# #### **Task 5: Create train/validate/test sets**

# Split data into training and testing sets, 80/20.

# In[19]:


# Split the data into training and testing sets
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Split the training set into training and validation sets, 75/25, to result in a final ratio of 60/20/20 for train/validate/test sets.

# In[20]:


# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=0)


# Confirm that the dimensions of the training, validation, and testing sets are in alignment.

# In[21]:


# Get shape of each training, validation, and testing set
X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape


# - The number of features (`11`) aligns between the training and testing sets.
# - The number of rows aligns between the features and the outcome variable for training (`11,450`) and both validation and testing data (`3,817`).

# ### **Task 6. Build models**

# ### **Build a random forest model**
# 

# Fit a random forest model to the training set. Use cross-validation to tune the hyperparameters and select the model that performs best on recall.

# In[30]:


# Instantiate the random forest classifier
rf = RandomForestClassifier(random_state=0)

# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [5, 7, None],
             'max_features': [0.3, 0.6],
            #  'max_features': 'auto'
             'max_samples': [0.7],
             'min_samples_leaf': [1,2],
             'min_samples_split': [2,3],
             'n_estimators': [75,100,200],
             }

# Define a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1'}

# Instantiate the GridSearchCV object
rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=5, refit='recall')


# Note this cell might take several minutes to run.

# In[31]:


get_ipython().run_cell_magic('time', '', 'rf_cv.fit(X_train_final, y_train)\n')


# In[32]:


# Examine best recall score
rf_cv.best_score_


# In[33]:


# Examine best parameters
rf_cv.best_params_


# **Exemplar response:**
# 
# This model performs exceptionally well, with an average recall score of 0.995 across the five cross-validation folds. After checking the precision score to be sure the model is not classifying all samples as claims, it is clear that this model is making almost perfect classifications.

# ### **Build an XGBoost model**

# In[34]:


# Instantiate the XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state=0)

# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [4,8,12],
             'min_child_weight': [3, 5],
             'learning_rate': [0.01, 0.1],
             'n_estimators': [300, 500]
             }

# Define a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1'}

# Instantiate the GridSearchCV object
xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=5, refit='recall')


# Note this cell might take several minutes to run.

# In[35]:


get_ipython().run_cell_magic('time', '', 'xgb_cv.fit(X_train_final, y_train)\n')


# In[36]:


xgb_cv.best_score_


# In[37]:


xgb_cv.best_params_


# **Exemplar response:**
# 
# This model also performs exceptionally well. Although its recall score is very slightly lower than the random forest model's, its precision score is perfect.

# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## **PACE: Execute**
# Consider the questions in your PACE Strategy Documentto reflect on the Execute stage.

# ### **Task 7. Evaluate models**
# 
# Evaluate models against validation data.
# 

# #### **Random forest**

# In[38]:


# Use the random forest "best estimator" model to get predictions on the validation set
y_pred = rf_cv.best_estimator_.predict(X_val_final)


# Display the predictions on the validation set.

# In[39]:


# Display the predictions on the validation set
y_pred


# 
# Display the true labels of the validation set.

# In[40]:


# Display the true labels of the validation set
y_val


# Create a confusion matrix to visualize the results of the classification model.

# In[41]:


# Create a confusion matrix to visualize the results of the classification model

# Compute values for confusion matrix
log_cm = confusion_matrix(y_val, y_pred)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=None)

# Plot confusion matrix
log_disp.plot()

# Display plot
plt.show()


# **Notes:**
# 
# The upper-left quadrant displays the number of true negatives: the number of opinions that the model accurately classified as so.
# 
# The upper-right quadrant displays the number of false positives: the number of opinions that the model misclassified as claims.
# 
# The lower-left quadrant displays the number of false negatives: the number of claims that the model misclassified as opinions.
# 
# The lower-right quadrant displays the number of true positives: the number of claims that the model accurately classified as so.
# 
# A perfect model would yield all true negatives and true positives, and no false negatives or false positives.
# 
# As the above confusion matrix shows, this model does not produce any false negatives.

# Create a classification report that includes precision, recall, f1-score, and accuracy metrics to evaluate the performance of the model.
# <br> </br>
# 

# In[42]:


# Create a classification report
# Create classification report for random forest model
target_labels = ['opinion', 'claim']
print(classification_report(y_val, y_pred, target_names=target_labels))


# The classification report above shows that the random forest model scores were nearly perfect. The confusion matrix indicates that there were 10 misclassifications&mdash;five false postives and five false negatives.

# #### **XGBoost**

# Now, evaluate the XGBoost model on the validation set.

# In[43]:


#Evaluate XGBoost model
y_pred = xgb_cv.best_estimator_.predict(X_val_final)


# In[44]:


y_pred


# In[45]:


# Compute values for confusion matrix
log_cm = confusion_matrix(y_val, y_pred)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=None)

# Plot confusion matrix
log_disp.plot()

# Display plot
plt.title('XGBoost - validation set');
plt.show()


# In[46]:


# Create a classification report
target_labels = ['opinion', 'claim']
print(classification_report(y_val, y_pred, target_names=target_labels))


# The results of the XGBoost model were also nearly perfect. However, its errors tended to be false negatives. Identifying claims was the priority, so it's important that the model be good at capturing all actual claim videos. The random forest model has a better recall score, and is therefore the champion model.

# ### **Use champion model to predict on test data**
# 
# Both random forest and XGBoost model architectures resulted in nearly perfect models. Nonetheless, in this case random forest performed a little bit better, so it is the champion model.
# 
# Now, use the champion model to predict on the test data.

# In[47]:


# Use champion model to predict on test data
y_pred = rf_cv.best_estimator_.predict(X_test_final)


# In[48]:


# Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=None)

# Plot confusion matrix
log_disp.plot()

# Display plot
plt.title('Random forest - test set');
plt.show()


# #### **Feature importances of champion model**
# 
# 

# In[49]:


importances = rf_cv.best_estimator_.feature_importances_
rf_importances = pd.Series(importances, index=X_test_final.columns)

fig, ax = plt.subplots()
rf_importances.plot.bar(ax=ax)
ax.set_title('Feature importances')
ax.set_ylabel('Mean decrease in impurity')
fig.tight_layout()


# The most predictive features all were related to engagement levels generated by the video. This is not unexpected, as analysis from prior EDA pointed to this conclusion.

# ### **Conclusion**
# 
# In this step use the results of the models above to formulate a conclusion. Consider the following questions:
# 
# 1. **Would you recommend using this model? Why or why not?**
# 
# 2. **What was your model doing? Can you explain how it was making predictions?**
# 
# 3. **Are there new features that you can engineer that might improve model performance?**
# 
# 4. **What features would you want to have that would likely improve the performance of your model?**
# 
# Remember, sometimes your data simply will not be predictive of your chosen target. This is common. Machine learning is a powerful tool, but it is not magic. If your data does not contain predictive signal, even the most complex algorithm will not be able to deliver consistent and accurate predictions. Do not be afraid to draw this conclusion.
# 
# 

# 
# 1. *Would you recommend using this model? Why or why not?*
# Yes, one can recommend this model because it performed well on both the validation and test holdout data. Furthermore, both precision and F<sub>1</sub> scores were consistently high. The model very successfully classified claims and opinions.
# </br>
# 2. *What was your model doing? Can you explain how it was making predictions?*
# The model's most predictive features were all related to the user engagement levels associated with each video. It was classifying videos based on how many views, likes, shares, and downloads they received.
# </br>
# 3. *Are there new features that you can engineer that might improve model performance?*
# Because the model currently performs nearly perfectly, there is no need to engineer any new features.
# </br>
# 4. *What features would you want to have that would likely improve the performance of your model?*
# The current version of the model does not need any new features. However, it would be helpful to have the number of times the video was reported. It would also be useful to have the total number of user reports for all videos posted by each author.

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
