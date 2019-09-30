#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:48:01 2019

@author: pivotalit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 11:47:21 2019

@author: pnagula
"""
import numpy as np
import pandas as pd
import random
import shutil
import pickle
import os
import sys
import re
import glob
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

path=sys.argv[1]

#def sentiment_analyzer_scores(analyser,sentence):
#    score = analyser.polarity_scores(sentence)
#    return score['pos'],score['neg'],score['neu'],score['compound']


def Extract_Feedback_Concept(ffile,cfiles):
    
    xl = pd.ExcelFile(ffile)
    feedback = xl.parse('National Issue')
    feedback['SNO'].replace('', np.nan, inplace=True)
    feedback.dropna(subset=['SNO'], inplace=True)
    for index, row in feedback.iterrows():
        strg=str(row['SNO'])
        strg=re.sub("[^0-9]", "",strg)
        feedback.at[index,'SNO']=strg
        strg=row['Feedback']
        strg=strg.replace("\n"," ")
        feedback.at[index,'Feedback']=strg
    feedback['Label'] = feedback['Label'].str.strip().str.lower()
    feedback.drop(['Category'],inplace=True,axis=1)    
    feedback['SNO'] = feedback['SNO'].astype('int32')
    print(str(len(feedback))+' rows read into Feedbacks dataframe from file '+feedback_file)
    concepts=None
    if cfiles:
       df3=pd.DataFrame()
       for concept_file in cfiles:
           concept_file=concept_file.replace('\n','')
           cxl = pd.ExcelFile(path+'/files/input_files/'+concept_file)
           df2 = cxl.parse('Feedback')
           df2['SNO'].replace('', np.nan, inplace=True)
           df2.dropna(subset=['SNO'], inplace=True)
           for index, row in df2.iterrows():
              strg=str(row['SNO'])
              strg=re.sub("[^0-9]", "",strg)
              df2.at[index,'SNO']=strg
           df2['SNO'] = df2['SNO'].astype('int32')
           print(str(len(df2))+' rows read into Concepts dataframe from file '+concept_file)
           df3=df3.append(df2,ignore_index=True)
  
       df3.drop(['Date'],inplace=True,axis=1)
       cols=list(df3)
       concepts=pd.DataFrame(columns=['SNO','Issue','Concept'])
       idx=0
       for row in df3.index:
           for column in cols:
               if df3.at[row,column]==1:
                  concepts.loc[idx]=[df3.at[row,'SNO'],df3.at[row,'Issue'],str(column).lower()] 
                  idx+=1
       concepts['SNO'] = concepts['SNO'].astype('int32') 
       print(str(len(concepts))+' concepts extracted from concept dataframe built from concept files ')
 
    return feedback, concepts

def Merge_Feedback_Concepts_Unmatched(feedback,concepts,final_df):
    if not concepts.empty:
       df3=pd.merge(feedback,concepts[['SNO','Concept']],on='SNO',how='left')
    else:
       df3=feedback 
    if not concepts.empty:
       df3=df3[df3['Concept'].isnull()]
       df3.drop(['Concept'],inplace=True,axis=1)
    print('Number of rows after left outer join Feedback files and Concepts files on SNO:',len(df3))
   
    for index, row in df3.iterrows():
        strg=row['Feedback']
        strg=strg.replace("\n"," ")
        df3.at[index,'Feedback']=strg
   
    df3['Label'] = df3['Label'].str.strip().str.lower()
    return df3, len(df3)

def Merge_Feedback_Concepts_Matched(feedback,concepts,final_df):
    df3=pd.merge(feedback,concepts[['SNO','Concept']],on='SNO',how='inner')
    df3['Label'] = df3['Label'].str.strip().str.lower()
    df3['Concept'] = df3['Concept'].str.strip().str.lower()
    print('Number of rows after inner joining Feedback files and Concepts files on SNO:',len(df3))
    return df3, len(df3)
        
#def Extract_Sentiment(feedback):
#    analyser = SentimentIntensityAnalyzer()
#    sentiment=np.array((0,0.0,0.0,0.0,0.0))
#    for index,row in feedback.iterrows():
#        pos,neg,neu,comp=sentiment_analyzer_scores(analyser,row['Feedback'])
#        sentiment=np.vstack((sentiment,np.column_stack((row['SNO'],pos,neg,neu,comp))))
#    sentiment=sentiment[1:,]
#    sentiment=pd.DataFrame(sentiment,columns=(['SNO','Positive','Negative','Neutral','Compound'])) 
#    sentiment['SNO']=sentiment['SNO'].astype('int32')
#    sentiment.to_csv(path+'/files/output_files/sentiment.tsv',sep='\t',index=False)
    
def Merge_Public_PA_data(final_df_matched,final_df_unmatched):
    
    turl=open(path+'/files/output_files/url_lists/all_train_copy.txt','a')
    vurl=open(path+'/files/output_files/url_lists/all_val_copy.txt','a')
    eurl=open(path+'/files/output_files/url_lists/all_test_copy.txt','w')

    if (not final_df_matched.empty) and sys.argv[2]=='training':
       train_set=random.sample(range(len(final_df_matched)-1),round((len(final_df_matched)-1)*.95))
       val_set=list(set(range(len(final_df_matched)-1))-set(train_set))

       for i in train_set:
       	   line=final_df_matched.loc[i]
           fname='pa_train_'+str(i)+'.story'
           turl.write('\n'+'pa_train_'+str(i))
           ofile=open(path+'/files/output_files/cnn/stories/'+fname,'w')
           ofile.write(line[1]+'\n\n')
           ofile.write('@highlight'+'\n')
           ofile.write(line[3])
           ofile.close()
    
       for i in val_set:
           line=final_df_matched.loc[i]
           fname='pa_val_'+str(i)+'.story'
           vurl.write('\n'+'pa_val_'+str(i))
           ofile=open(path+'/files/output_files/cnn/stories/'+fname,'w')
           ofile.write(line[1]+'\n\n')
           ofile.write('@highlight'+'\n')
           ofile.write(line[3])
           ofile.close()
    if sys.argv[2]=='scoring':    
       for index,row in final_df_unmatched.iterrows():  
           line=final_df_unmatched.loc[index]
           fname='pa_test_U_'+str(index)+'.story'
           eurl.write('pa_test_U_'+str(index)+'\n')
           ofile=open(path+'/files/output_files/cnn/stories/'+fname,'w')
           ofile.write(line[1]+'\n\n')
           ofile.write('@highlight'+'\n')
           ofile.write('Concept to be generated')
           ofile.close()        
    
    turl.close()
    vurl.close()
    eurl.close()
        
if __name__ == '__main__':
   matched=path+'/files/output_files/matched_feedback_concepts.tsv'
   unmatched=path+'/files/output_files/unmatched_labels.tsv'
   if os.path.exists(matched):
      os.remove(matched)
   if os.path.exists(unmatched):
      os.remove(unmatched)
   
   f=open(path+'/files/input_files/file_locations.tsv','r')
   files=f.readlines()
   f.close()
   final_df_matched = pd.DataFrame()
   final_df_unmatched = pd.DataFrame()
   feedback=pd.DataFrame()
   concepts=pd.DataFrame()
#   
   print('####### Extracting Feedback text and concepts from excel files #######')
   for file in files:
       recs=len(file.split('\t'))
       feedback_file=file.split('\t')[0]
       feedback_file=feedback_file.replace('\n','')
       if recs > 1:	
          concept_files=file.split('\t')[1].split(',')      
       else:
          concept_files=None
       fb, cp = Extract_Feedback_Concept(path+'/files/input_files/'+feedback_file,concept_files)
       feedback=feedback.append(fb,ignore_index=True)
       if concept_files:
          concepts=concepts.append(cp,ignore_index=True)
   print('-----------------------------------------------------------------------------')
   print('Total '+str(len(feedback))+' feedback text and '+str(len(concepts))+' concepts')
   print('-----------------------------------------------------------------------------')
  
   print('###### Pulling all feedback text that DON''T have concepts generated ########')
   final_df_unmatched, nrows=Merge_Feedback_Concepts_Unmatched(feedback,concepts,final_df_unmatched)
   
   if not concepts.empty:
      print('###### Pulling all feedback text that have concepts ########')
      final_df_matched, nrows=Merge_Feedback_Concepts_Matched(feedback,concepts,final_df_matched)
   
   print('--------------------------------------------------------------------')
   print('Concepts to be generated from '+str(len(final_df_unmatched))+' feedbacks ')
   print('--------------------------------------------------------------------')
  # print('Extracting Sentiment and write to an output file.....')
  # Extract_Sentiment(feedback)  
   
   final_df_matched.to_csv(matched,sep='\t',header=False)
   final_df_unmatched.to_csv(unmatched,sep='\t',header=False)

   shutil.copyfile(path+'/files/static_data/url_lists/all_train.txt',path+'/files/output_files/url_lists/all_train_copy.txt')
   shutil.copyfile(path+'/files/static_data/url_lists/all_val.txt',path+'/files/output_files/url_lists/all_val_copy.txt')
   shutil.copyfile(path+'/files/static_data/url_lists/all_test.txt',path+'/files/output_files/url_lists/all_test_copy.txt')
   if os.path.exists(path+'/files/output_files/cnn/stories/'):
      shutil.rmtree(path+'/files/output_files/cnn/stories/')
   if os.path.exists(path+'/files/output_files/dailymail/stories/'):   
      shutil.rmtree(path+'/files/output_files/dailymail/stories/')       
   if sys.argv[2]=='scoring':
      ofile=open(path+'/files/output_files/url_lists/all_train_copy.txt','w')
      ofile.close()
      ofile=open(path+'/files/output_files/url_lists/all_val_copy.txt','w')
      ofile.close()
      os.mkdir(path+'/files/output_files/cnn/stories/')
      os.mkdir(path+'/files/output_files/dailymail/stories/')
   else:      
      shutil.copytree(path+'/files/static_data/cnn/stories/',path+'/files/output_files/cnn/stories/')
      shutil.copytree(path+'/files/static_data/dailymail/stories/',path+'/files/output_files/dailymail/stories/')      
   
   Merge_Public_PA_data(final_df_matched,final_df_unmatched)
# 

   
