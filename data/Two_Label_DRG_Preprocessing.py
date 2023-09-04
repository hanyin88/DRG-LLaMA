import pandas as pd
import numpy as np
import re
import json



def drg_dissection(drg_34_path, train_set_path, test_set_path, id2label_path):

  drg_34_dissection = pd.read_csv(drg_34_path, sep="\t", header=0)

  # only keep the column on DRG and Description
  drg_34_dissection = drg_34_dissection[["DRG", "Description"]]

  # Make a new column called CC/MCC, where the value is 1 if the value in column Description contains "WITH CC" or "W CC"
  #  2 if the value in column Description contains "WITH MCC" or "W MCC", 0 if the value in column Description contains "WITHOUT CC/MCC" or "W/O CC/MCC"
  # 3 if the value in column Description contains "WITHOUT MCC"or "W/O MCC", else 4 (which essentially represents not applicable)
    # Note, in the approach here, with cc/mcc will be classified with cc

  drg_34_dissection["CC/MCC"] = drg_34_dissection["Description"].apply(lambda x: 1 if ("WITH CC" in x) or ("W CC" in x) else
                                                                        (2 if ("WITH MCC" in x) or ("W MCC" in x) else
                                                                          (0 if ("WITHOUT CC/MCC" in x) or ("W/O CC/MCC" in x) else
                                                                            (3 if ("WITHOUT MCC" in x) or ("W/O MCC" in x) else 4))))



  # Make a new collumn called principal_diagnosis, which uses regex to extraxt text before one of the following words:
  # "WITH CC", "W CC", "WITH MCC", "W MCC", "WITHOUT CC/MCC", "W/O CC/MCC", "WITHOUT MCC", "W/O MCC"

  pc_patterh = r"^(.*?)(?:WITH CC|W CC|WITH MCC|W MCC|WITHOUT CC/MCC|W/O CC/MCC|WITHOUT MCC|W/O MCC)"
  def find_pc(note):
    if re.search(pc_patterh, note):
      return re.search(pc_patterh, note).group(1)
    else:
      return note

  drg_34_dissection["principal_diagnosis"] = drg_34_dissection["Description"].apply(find_pc)

  #In the column of principal_diagnosis, clean up some typo and inconsistence in the offical DRG 34 table

  pattern_proc = '|'.join(["PROEDURESC ", "PROEDURESC ", "PROCEDURSE ", "PROC ", "PROCEDURE "])

  drg_34_dissection["principal_diagnosis"] = drg_34_dissection["principal_diagnosis"].str.replace(pattern_proc,"PROCEDURES ", regex=True)
  drg_34_dissection["principal_diagnosis"] = drg_34_dissection["principal_diagnosis"].str.replace("BILIARY TRACT PROCEDURES EXCEPT ONLY CHOLECYST ", "BILIARY TRACT PROCEDURES EXCEPT ONLY CHOLECYSTECTOMY ", regex=True)
  drg_34_dissection["principal_diagnosis"] = drg_34_dissection["principal_diagnosis"].str.replace("CATHETERATION","CATHETERIZATION", regex=True)
  drg_34_dissection["principal_diagnosis"] = drg_34_dissection["principal_diagnosis"].str.replace("GASTROENTERISTIS","GASTROENTERITIS", regex=True)
  drg_34_dissection["principal_diagnosis"] = drg_34_dissection["principal_diagnosis"].str.replace("FIXATIOM","FIXATION", regex=True)
  drg_34_dissection["principal_diagnosis"] = drg_34_dissection["principal_diagnosis"].str.replace("CHEMOTHERPY","CHEMOTHERAPY", regex=True)
  drg_34_dissection["principal_diagnosis"] = drg_34_dissection["principal_diagnosis"].str.replace("REMOVAL INTERNAL","REMOVAL OF INTERNAL", regex=True)
  drg_34_dissection["principal_diagnosis"] = drg_34_dissection["principal_diagnosis"].str.replace("CHEMOTHERAPY WITH ACUTE LEUKEMIA AS SDX OR WITH HIGH DOSE CHEMOTHERAPY AGENT","CHEMOTHERAPY WITH ACUTE LEUKEMIA AS SDX", regex=True)



  # Number of unique principal_diagnosis: 340
  PC_count = drg_34_dissection.principal_diagnosis.nunique()
  CC_count = 5

  # Make a new collum called principal_diagnosis_lable, which is an interger from 0 to 340
  drg_34_dissection["principal_diagnosis_lable"] = drg_34_dissection["principal_diagnosis"].map(dict(zip(drg_34_dissection.principal_diagnosis.drop_duplicates(), range(0, PC_count))))

  def to_categorical(y, num_classes):
      """ 1-hot encodes a tensor """
      return np.eye(num_classes, dtype='uint8')[y]

  # Make a new collum called multi_label, which is a one hot vector with 345 elements
  # The first 340 (0-339) elements are the one hot vector for principal_diagnosis_lable
  # The 340th to 344th (340-344) elements are the one hot vector for CC/MCC
  drg_34_dissection["multi_label"] = drg_34_dissection.apply(lambda x: np.concatenate((to_categorical(x["principal_diagnosis_lable"], num_classes=PC_count), to_categorical(x["CC/MCC"], num_classes=CC_count))), axis=1)

  # make a new collum called two_label, which is a list of two elements: principal_diagnosis_lable and CC/MCC
  drg_34_dissection["two_label"] = drg_34_dissection.apply(lambda x: [x["principal_diagnosis_lable"], x["CC/MCC"]], axis=1)

  ##make database
  train = pd.read_csv(train_set_path)
  test = pd.read_csv(test_set_path)

  # read id to label mapping
  id2label = pd.read_csv(id2label_path)

  # in train and test, add corresponding drg_34_code where label match the label in id2label
  train = train.merge(id2label, on="label", how="left")
  test = test.merge(id2label, on="label", how="left")

  # in train and test, add column of multi_label where drg_34_code match the drg_34_code in drg_34_dissection
  train = train.merge(drg_34_dissection[["DRG", "multi_label"]], left_on="drg_34_code", right_on="DRG", how="left")
  test = test.merge(drg_34_dissection[["DRG", "multi_label"]], left_on="drg_34_code", right_on="DRG", how="left")


  # for train and test, only keep text and multi_label and rename multi_label to label
  train = train[["text", "multi_label"]]
  train = train.rename(columns={"multi_label": "label"})
  test = test[["text", "multi_label"]]
  test = test.rename(columns={"multi_label": "label"})

  return train, test, drg_34_dissection

if __name__ == "__main__":
    # Read path from the json file
  with open('paths.json', 'r') as f:
      path = json.load(f)
      drg_34_path = path["drg_34_path"]
      drg_34_dissection_path = path["drg_34_dissection_path"]
      train_set_path = path["train_set_path"]
      test_set_path = path["test_set_path"]
      id2label_path = path["id2label_path"]
      multi_train_set_path = path["multi_train_set_path"]
      multi_test_set_path = path["multi_test_set_path"]

  train, test, drg_34_dissection = drg_dissection(drg_34_path, train_set_path, test_set_path, id2label_path)

  train.to_csv(multi_train_set_path, index=False)
  test.to_csv(multi_test_set_path, index=False)
  drg_34_dissection.to_csv(drg_34_dissection_path, index=False)