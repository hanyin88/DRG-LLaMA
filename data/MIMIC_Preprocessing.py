import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
    
##############Step 1: Extract "brief hospital course" from discharge summary#####################

def extract_HC(dc_summary_path):

  # Load the data
  dc_summary_raw = pd.read_csv(dc_summary_path)

  # Set up the regular expression to extract hospital course from discharge summary
  # Of note these patterns would not caputre all hospital courses, and is indeed a convservative approach to ensure quality of data
  pattern1  = re.compile("Brief Hospital Course.*\n*((?:\n.*)+?)(Medications on Admission|___  on Admission|___ on Admission)")
  pattern2  = re.compile("Brief Hospital Course.*\n*((?:\n.*)+?)Discharge Medications")
  pattern3  = re.compile("(Brief Hospital Course|rief Hospital Course|HOSPITAL COURSE)\
                        .*\n*((?:\n.*)+?)\
                        (Medications on Admission|Discharge Medications|DISCHARGE MEDICATIONS|DISCHARGE DIAGNOSIS|Discharge Disposition|___ Disposition|CONDITION ON DISCHARGE|DISCHARGE INSTRUCTIONS)")
  pattern4  = re.compile("(Mg-[12].|LACTATE-[12].|Epi-|Gap-|COUNT-|TRF-)___(.*\n*((?:\n.*)+?))(Medications on Admission)")


  # Idea here is to try more convservaite pattern first, if not work, try less conservative pattern
  def split_note(note):
    if re.search(pattern1, note):
      return re.search(pattern1, note).group(1)
    else:
      if re.search(pattern2, note):
        return re.search(pattern2, note).group(1)
      else:
        if re.search(pattern3, note):
          return re.search(pattern3, note).group(2)
        else:
          if re.search(pattern4, note):
            return re.search(pattern4, note).group(2)
          else:
            return None

  # Apply the function to dc_summary_raw to extract hospital course
  dc_summary_raw["hospital_course"] = dc_summary_raw["text"].apply(split_note)

  # Drop those records that do not have hospital course captured with above regular expression patterns
  dc_summary = dc_summary_raw[["hadm_id", "hospital_course"]].dropna()

  # Get the number of words for each hospital course. Note that the current method is not accurate due to presense of special characters, but it's good enough for our purpose
  dc_summary["num_words"] = dc_summary["hospital_course"].apply(lambda x: len(x.split()))

  # Remove the notes with less than 40 words
  dc_summary = dc_summary[dc_summary["num_words"] > 40]

  # Remove duplicate hospital courses (but keep the first one), as most of these notes represent low quality data
  dc_summary = dc_summary.drop_duplicates(subset=["hospital_course"], keep="first")

  # Mean number of words in the hospital course is 378
  dc_summary["num_words"].mean()

  # only keep hadm_id and hospital_course
  dc_summary = dc_summary[["hadm_id", "hospital_course"]]

  return dc_summary

##############Step 2: Map all DRG codes to MS-DRG v34.0#####################
# HCFA DRG is the MS-DRG code (https://github.com/MIT-LCP/mimic-code/issues/1561)
def map_drg(mimic_drg_path, drg_34_path, my_mapping_path):

  drg = pd.read_csv(mimic_drg_path)

  drg = drg[["hadm_id", "drg_code", "drg_type", "description"]][drg["drg_type"] == "HCFA"]

  # We mapped all MS-DRG codes to v.34, which was released in 2016
  # Read the DRG v.34.0 codes from the csv file
  drg_34 = pd.read_csv(drg_34_path, sep="\t", header=0)

  # Extra the set of all DRG description mentioned in the dataset
  drg_mapping = pd.DataFrame(drg["description"].drop_duplicates())

  # Create a second column called tranformation, which is to make basic normalizaiton of the DRG descriptions (e.g., W to WITH, W/O to WITHOUT, & to AND)
  drg_mapping["transformation"] = drg_mapping["description"].str.replace("W/O", "WITHOUT").str.replace(" W ", " WITH ").str.replace("&", "AND")
  drg_mapping["transformation"] = drg_mapping["transformation"].str.replace(",", "").str.replace(" CATH ", " CATHETERIZATION ").str.replace(" PROC ", " PROCEDURES ")

  # Read the mapping rule to MS-DRG v.34
  my_mapping = pd.read_csv(my_mapping_path, header=0)

  # create a dictionay from my_mapping, where raw_description is the key and DRG_34_description is the value
  my_mapping_dict = dict(zip(my_mapping.raw_description, my_mapping.DRG_34_description))

  # Create a thrid column called drg_34_description, which copy the transformation column if the description is in drg_34
  # otherwide copy the value from my_mapping_dict where the key is the transformation
  drg_mapping["drg_34_description"] = drg_mapping["transformation"].where(drg_mapping["transformation"].isin(drg_34.Description), other=drg_mapping["transformation"].map(my_mapping_dict))

  # check number of na in drg_34_description: 20
  drg_mapping.drg_34_description.isna().sum()

  #rename the column name of description to raw_description
  drg_mapping = drg_mapping.rename(columns={"description": "raw_description"})

  # Crate a table called drg_code_mapping by joining drg_mapping and drg_34 by drg_34_description
  drg_code_mapping = pd.merge(drg_mapping, drg_34, how="left", left_on="drg_34_description", right_on="Description")

  # make a dictinoary from drg_code_mapping, where raw_description is the key and DRG is the value
  drg_mapping_dict = dict(zip(drg_code_mapping.raw_description, drg_code_mapping.DRG))

  # In the table drg, create a new column called drg_34_description, which is the mapped value from drg_mapping_dict where the key is description
  drg["drg_34_code"] = drg["description"].map(drg_mapping_dict)

  # drop the rows with na in drg_34_code
  drg = drg.dropna(subset=["drg_34_code"])

  # only keep hadm_id and drg_34_code, and change drg_34_code to int
  drg = drg[["hadm_id", "drg_34_code"]].astype({"drg_34_code": int})

  return drg

################Step 3. Merge discharge summary and drg, and split into training/testing sets#####################
#merge drg and dc_summary by hadm_id 
def merge_HC_drg(dc_summary, drg):

  dc_drg = pd.merge(dc_summary, drg, how="inner", on="hadm_id")

  # remove drg_34_code with less than 2 observations
  # in this step removed code 998, 985 and 793
  dc_drg = dc_drg.groupby("drg_34_code").filter(lambda x: len(x) >= 2)


  # number of unique drg_34_code in dc_drg: 738
  drg_count = dc_drg.drg_34_code.nunique()


  # rank dc_drg by drg_34_code
  dc_drg = dc_drg.sort_values(by=["drg_34_code"])

  # make a new dataframe called id2label, where the first column is drg_34_code and the second column is the rank of drg_34_code starting form 0 to 737
  id2label = pd.DataFrame(dc_drg.drg_34_code.drop_duplicates())
  id2label["label"] = range(0, drg_count)

  # in dc_drg, create a new column called label, which is the mapped value from id2label where the key is drg_34_code
  dc_drg["label"] = dc_drg["drg_34_code"].map(dict(zip(id2label.drg_34_code, id2label.label)))

  # split dc_drc into train and test, test takes 10% of the data, set radoom state to 42, stratify by label
  train, test = train_test_split(dc_drg, test_size=0.1, random_state=42, stratify=dc_drg.label)

  # rename hospital_course to text, remove column of hadm_id and drg_34_code, and save train and test to csv
  train = train.rename(columns={"hospital_course": "text"})
  test = test.rename(columns={"hospital_course": "text"})
  train = train[["text", "label"]]
  test = test[["text", "label"]]

  return train, test, id2label

if __name__ == "__main__":
    # Read path from the json file
  with open('paths.json', 'r') as f:
      path = json.load(f)
      dc_summary_path = path["dc_summary_path"]
      mimic_drg_path = path["mimic_drg_path"]
      drg_34_path = path["drg_34_path"]
      my_mapping_path = path["my_mapping_path"]
      train_set_path = path["train_set_path"]
      test_set_path = path["test_set_path"]
      id2label_path = path["id2label_path"]

  
  drg = map_drg(mimic_drg_path, drg_34_path, my_mapping_path)
  dc_summary = extract_HC(dc_summary_path)
  
  train, test, id2label = merge_HC_drg(dc_summary, drg)

  id2label.to_csv(id2label_path, index=False)
  train.to_csv(train_set_path, index=False)
  test.to_csv(test_set_path, index=False)
