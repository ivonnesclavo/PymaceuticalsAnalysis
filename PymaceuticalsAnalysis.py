#!/usr/bin/env python
# coding: utf-8

# # Pymaceuticals Inc.
# ---
# 
# ### Observations:
# 
# -The Drug Regimen with the lowest average tumor volume is Ramicane and the highest is Ketapril.
# 
# -The mice observed were 51% male and 49% female.
# 
# -The correlation between mouse weight and average tumor volume was 0.83, which indicates a strong positive relationship between the two variables. We can infer that heavier mice will have larger average tumor volumes and if mouse weight increases, so will the average tumor volume.
# 
#  

# In[24]:


# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from scipy.stats import linregress

# Study data files
mouse_metadata_path = "Starter_Code/Pymaceuticals/data/Mouse_metadata.csv"
study_results_path = "Starter_Code/Pymaceuticals/data/Study_results.csv"

# Read the mouse data and the study results
mouse_metadata = pd.read_csv(mouse_metadata_path)
study_results = pd.read_csv(study_results_path)

# Combine the data into a single DataFrame
pymaceuticals_data = pd.merge(study_results,mouse_metadata,on = "Mouse ID", how= "left")

# Display the data table for preview
pymaceuticals_data.head()


# In[2]:


# Checking the number of mice.
mouse_count = pymaceuticals_data["Mouse ID"].nunique()
mouse_count


# In[3]:


# Our data should be uniquely identified by Mouse ID and Timepoint
# Get the duplicate mice by ID number that shows up for Mouse ID and Timepoint. 
duplicate_mouse_id = pymaceuticals_data[pymaceuticals_data.duplicated(subset=["Mouse ID", "Timepoint"])]["Mouse ID"].unique()
duplicate_mouse_id


# In[4]:


# Optional: Get all the data for the duplicate mouse ID. 
g989_data = pymaceuticals_data[pymaceuticals_data["Mouse ID"] == "g989"]
g989_data


# In[5]:


# Create a clean DataFrame by dropping the duplicate mouse by its ID.
clean_pymaceuticals = pymaceuticals_data[pymaceuticals_data["Mouse ID"] != "g989"]
clean_pymaceuticals


# In[6]:


# Checking the number of mice in the clean DataFrame.
clean_mouse_count = clean_pymaceuticals["Mouse ID"].nunique()
clean_mouse_count


# ## Summary Statistics

# In[7]:


# Generate a summary statistics table of mean, median, variance, standard deviation, and SEM of the tumor volume for each regimen
# Use groupby and summary statistical methods to calculate the following properties of each drug regimen: 
# mean, median, variance, standard deviation, and SEM of the tumor volume. 

mean_tumor_volume_groupedby_drugregimen = clean_pymaceuticals.groupby("Drug Regimen").mean()["Tumor Volume (mm3)"]
median_tumor_volume_groupedby_drugregimen = clean_pymaceuticals.groupby("Drug Regimen").median()["Tumor Volume (mm3)"]
var_tumor_volume_groupedby_drugregimen = clean_pymaceuticals.groupby("Drug Regimen").var()["Tumor Volume (mm3)"]
std_tumor_volume_groupedby_drugregimen = clean_pymaceuticals.groupby("Drug Regimen").std()["Tumor Volume (mm3)"]
sem_tumor_volume_groupedby_drugregimen = clean_pymaceuticals.groupby("Drug Regimen").sem()["Tumor Volume (mm3)"]

# Assemble the resulting series into a single summary DataFrame.
summary_df = pd.DataFrame({
    "Mean": mean_tumor_volume_groupedby_drugregimen,
    "Median": median_tumor_volume_groupedby_drugregimen,
    "Variance": var_tumor_volume_groupedby_drugregimen,
    "Std. Dev": std_tumor_volume_groupedby_drugregimen,
    "SEM":sem_tumor_volume_groupedby_drugregimen
})
summary_df.name = "Summary Statistics of Tumor Volume grouped by Drug Regimen"

print(summary_df.name)
summary_df


# In[8]:


# A more advanced method to generate a summary statistics table of mean, median, variance, standard deviation,
# and SEM of the tumor volume for each regimen (only one method is required in the solution)

# Using the aggregation method, produce the same summary statistics in a single line
summary_df_2 =clean_pymaceuticals.groupby("Drug Regimen").agg({"Tumor Volume (mm3)":["mean","median","var","std","sem"]})
summary_df_2


# ## Bar and Pie Charts

# In[9]:


# Generate a bar plot showing the total number of rows (Mouse ID/Timepoints) for each drug regimen using Pandas.

mouse_per_drugregimen_pandas = clean_pymaceuticals["Drug Regimen"].value_counts()
mouse_per_drugregimen_pandas.plot(kind="bar", title = "Total number of Mouse ID/Timepoints for each Drug Regimen")
plt.xlabel("Drug Regimen")
plt.ylabel("# of Observed Mouse Timepoints")


# In[10]:


# Generate a bar plot showing the total number of rows (Mouse ID/Timepoints) for each drug regimen using pyplot.
mouse_per_drugregimen_pyplot = clean_pymaceuticals["Drug Regimen"].value_counts()
plt.bar(mouse_per_drugregimen_pyplot.index,mouse_per_drugregimen_pyplot.values)
plt.xlabel("Drug Regimen")
plt.ylabel("# of Observed Mouse Timepoints")
plt.xticks(rotation=90)
plt.show()


# In[11]:


# Generate a pie plot showing the distribution of female versus male mice using Pandas
gender_count = clean_pymaceuticals["Sex"].value_counts()
colors=["deepskyblue","hotpink"]
gender_count.plot(kind="pie",autopct='%1.1f%%',colors=colors)
plt.show()


# In[12]:


# Generate a pie plot showing the distribution of female versus male mice using pyplot
gender_count = clean_pymaceuticals["Sex"].value_counts()
colors=["deepskyblue","hotpink"]
plt.pie(gender_count, labels=gender_count.index, autopct='%1.1f%%',colors=colors)
plt.title("Sex")
plt.show()


# ## Quartiles, Outliers and Boxplots

# In[13]:


# Calculate the final tumor volume of each mouse across four of the treatment regimens:  
# Capomulin, Ramicane, Infubinol, and Ceftamin
# Start by getting the last (greatest) timepoint for each mouse
max_tumor =clean_pymaceuticals.groupby(["Mouse ID"])["Timepoint"].max()
max_tumor = max_tumor.reset_index()
# Merge this group df with the original DataFrame to get the tumor volume at the last timepoint
merged_tumor = max_tumor.merge(clean_pymaceuticals, on = ["Mouse ID", "Timepoint"],how="left")
merged_tumor


# In[14]:


# Put treatments into a list for for loop (and later for plot labels)
treatments = ["Capomulin", "Ramicane", "Infubinol", "Ceftamin"]

# Create empty list to fill with tumor vol data (for plotting)
tumor_vol_list =[]

# Calculate the IQR and quantitatively determine if there are any potential outliers. 
for drug in treatments:
    
    # Locate the rows which contain mice on each drug and get the tumor volumes
    tumor_volume = merged_tumor.loc[merged_tumor["Drug Regimen"] == drug, "Tumor Volume (mm3)"]
    
    # add subset 
    tumor_vol_list.append(tumor_volume)
    
    # Determine outliers using upper and lower bounds
    quartiles = tumor_volume.quantile([.25,.5,.75])
    lowerq=quartiles[0.25]
    upperq=quartiles[0.75]
    iqr = upperq -lowerq
    lower_bound = lowerq - (1.5 * iqr)
    upper_bound = upperq + (1.5 * iqr)
    outliers = tumor_volume.loc[(tumor_volume <lower_bound)| (tumor_volume > upper_bound)]
    print(f"{drug}'s potential outliers are : {outliers}")


# In[15]:


# Generate a box plot that shows the distrubution of the tumor volume for each treatment group.
plt.boxplot(tumor_vol_list,labels=treatments)
plt.title("Tumor Volume Distribution per Drug Regimen")
plt.show()


# ## Line and Scatter Plots

# In[16]:


# Generate a line plot of tumor volume vs. time point for a single mouse treated with Capomulin
capomulin = clean_pymaceuticals[clean_pymaceuticals["Drug Regimen"] == "Capomulin"]
capomulin_j246 = capomulin[capomulin["Mouse ID"] == "j246"]

plt.plot(capomulin_j246["Timepoint"], capomulin_j246["Tumor Volume (mm3)"], color = "purple", marker ="o")
plt.xlabel("Timepoint")
plt.ylabel("Tumor Volume (mm3)")
plt.title("Tumor Volume vs. Time Point for Mouse j246 (Capomulin)")
plt.show()


# In[17]:


# Generate a scatter plot of mouse weight vs. the average observed tumor volume for the entire Capomulin regimen
capomulin = clean_pymaceuticals[clean_pymaceuticals["Drug Regimen"] == "Capomulin"]
average_tumor_volume = capomulin.groupby("Mouse ID")["Tumor Volume (mm3)"].mean()
merged_data = pd.merge(average_tumor_volume, capomulin[["Mouse ID", "Weight (g)"]], on="Mouse ID", how="inner")
plt.scatter(merged_data["Weight (g)"], merged_data["Tumor Volume (mm3)"], marker='o')
plt.xlabel("Weight")
plt.ylabel("Average Tumor Volume (mm3)")
plt.title("Mouse Weight vs. Average Tumor Volume (Capomulin)")


# ## Correlation and Regression

# In[28]:


# Calculate the correlation coefficient and a linear regression model 
# for mouse weight and average observed tumor volume for the entire Capomulin regimen
correlation = round(st.pearsonr(merged_data["Weight (g)"],merged_data["Tumor Volume (mm3)"])[0],2)
print(f"The correlation between mouse weight and average tumor volume is {correlation}")

slope, intercept, r_value, p_value, std_err = linregress(merged_data["Weight (g)"], merged_data["Tumor Volume (mm3)"])
regress_values = merged_data["Weight (g)"] * slope + intercept
lin_eq = "y = " + str(round(slope,2))+ "x + " + str(round(intercept,2))

plt.scatter(merged_data["Weight (g)"], merged_data["Tumor Volume (mm3)"], marker='o', edgecolors='black')
plt.xlabel("Mouse Weight (g)")
plt.ylabel("Average Tumor Volume (mm3)")
plt.title("Mouse Weight vs. Average Tumor Volume (Capomulin Regimen)")

plt.plot(merged_data["Weight (g)"], intercept + slope * merged_data["Weight (g)"], color='red')


# In[ ]:




