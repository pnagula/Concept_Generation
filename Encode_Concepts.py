#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:33:35 2019

@author: pivotalit
"""
import glob
import os
import pandas as pd
fileList = glob.glob('/users/pivotalit/downloads/PreparationWork/decoded/firstpgexp_400/*_decoded.txt')
df=pd.DataFrame(columns=['Index','Concept'])
for filePath in fileList:
    f=open(filePath,'r')
    index=int(os.path.basename(os.path.normpath(filePath)).split('_')[0])
    concept_starts=False
    for line in f:
        if concept_starts:
           df=df.append({'Index':index,'Concept':line},ignore_index=True)
        if 'Generated Concept:' in line:
           concept_starts=True 
    f.close()
file='/users/pivotalit/downloads/PreparationWork/unmatched_clustered_labels.tsv'
unmatched=pd.read_csv(file,sep='\t',header=None) 
unmatched.columns=['Index','Label','Cluster','SNO','Summarized_Feedback']       
unmatched=pd.merge(unmatched,df,how='inner',on='Index')
ofile='/users/pivotalit/downloads/PreparationWork/concepts_generated_firstpgexp_400.csv'
unmatched.to_csv(ofile,sep='\t',index=False)