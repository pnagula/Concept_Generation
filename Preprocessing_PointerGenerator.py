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
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import pickle
import os
import re
import glob
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.summarization.summarizer import summarize

path='/users/pivotalit/downloads/PreparationWork/'

def sentiment_analyzer_scores(analyser,sentence):
    score = analyser.polarity_scores(sentence)
    return score['pos'],score['neg'],score['neu'],score['compound']
    
def KMeans_Clustering(dist):
    distortions=[]
    inertia=[]
    silhouette=[]
    if dist.shape[0] < 15:
       ubound=dist.shape[0]
    else:
       ubound=15
    for j in range(2,ubound,1):
        km = KMeans(n_clusters=j)
        clusters=km.fit_predict(dist)
        silhouette.append(metrics.silhouette_score(dist, clusters))
        distortions.append(sum(np.min(cdist(dist, km.cluster_centers_, 'euclidean'), axis=1)) / dist.shape[0])
        inertia.append(km.inertia_)
    return distortions,inertia,np.max(silhouette)

def Doc2Vec_Model(idx,docs,documents1,sent_np,mincount,ep):
    model = Doc2Vec(documents1, vector_size=300, window=2, min_count=mincount, workers=4,dm=1,epochs=ep)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    x=[]
    x=[model.infer_vector(x1) for x1 in docs[idx]]
    vect_array=x[0]
    firstime=True
    for i in x:
        if firstime:
           firstime=False
        else:
           vect_array=np.vstack((vect_array,i))

    features=np.concatenate((sent_np[1:,],vect_array),axis=1)
    return features

def Save_Plot(mds,dist,Label,clusters):
    pos=mds.fit_transform(dist)
    xs, ys = pos[:, 0], pos[:, 1]
    plotdf = pd.DataFrame(dict(x=xs, y=ys, label=clusters)) 
    groups = plotdf.groupby('label')
    fig, ax = plt.subplots(figsize=(17, 9)) # set size
    ax.margins(0.05) #Optional, just adds 5% padding to the autoscaling
    cluster_names = {0: 'Cluster 0',1: 'Cluster 1',2: 'Cluster 2',3:'Cluster 3',4: 'Cluster 4',5: 'Cluster 5',6: 'Cluster 6',7: 'Cluster 7',8: 'Cluster 8'}
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e',5: '#B0C4DE',6: '#DCDCDC',7: '#FFD700',8: '#778899'}
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
        ax.set_aspect('auto')
        ax.tick_params(\
                       axis= 'x',          # changes apply to the x-axis
                       which='both',      # both major and minor ticks are affected
                       bottom='off',      # ticks along the bottom edge are off
                       top='off',         # ticks along the top edge are off
                       labelbottom='off')
        ax.tick_params(\
                       axis= 'y',         # changes apply to the y-axis
                       which='both',      # both major and minor ticks are affected
                       left='off',      # ticks along the bottom edge are off
                       top='off',         # ticks along the top edge are off
                       labelleft='off')
    ax.legend(numpoints=1)
    plt.savefig('/users/pivotalit/downloads/PreparationWork/clusters_plots/'+Label.replace(' ','')+'.png')

def Cluster_Unmatched_Feedback(final_df_unmatched,mds,saveplot):
    analyser = SentimentIntensityAnalyzer()
    docs=[[word_tokenize(x1.lower())  for x1 in x] for x in final_df_unmatched['Feedback']]
    outdf=pd.DataFrame(columns = ['SNO','Label','Feedback','Concept'])
    epocs=[10,15,20,25,30]

    for idx,Label,SNO,Feedback in final_df_unmatched.itertuples():
        if len(Feedback) >= 5:
            sent_np=np.array((0.0,0.0,0.0,0.0))
            Feedback=final_df_unmatched['Feedback'][idx]
            for feed in Feedback:
                pos,neg,neu,comp=sentiment_analyzer_scores(analyser,feed)
                sent_np=np.vstack((sent_np,np.column_stack((pos,neg,neu,comp))))
               
            documents1 = [TaggedDocument(doc, [i])  for i, doc in enumerate(docs[idx]) ]
            mincount=int(round(len(Feedback)*.05))
            sil=[]
            for ep in epocs:
                features=Doc2Vec_Model(idx,docs,documents1,sent_np,mincount,ep)
                dist=1-cosine_similarity(features)       
                distortions,inertia,silhouette=KMeans_Clustering(dist) 
                sil.append(silhouette)

            epochs=epocs[np.argmax(sil)]
            features=Doc2Vec_Model(idx,docs,documents1,sent_np,mincount,epochs)
            dist=1-cosine_similarity(features)       
            distortions,inertia,silhouette=KMeans_Clustering(dist) 

            first_order=np.gradient(inertia)
            second_order=np.gradient(first_order)
            number_of_clusters=np.argmax(second_order)+2
            km = KMeans(n_clusters=number_of_clusters)
            clusters=km.fit_predict(dist)
            if saveplot:
               Save_Plot(mds,dist,Label,clusters) 
        else:
            clusters=[0]
        for sno,feed,clust in zip(SNO,Feedback,clusters):
            outdf=outdf.append({'SNO':sno,'Label':Label,'Feedback':feed,'Concept':'cluster_'+str(clust)},ignore_index=True)
    #final_file='/users/pivotalit/downloads/PreparationWork/issues_clustered.xlsx'
    #outdf.to_excel(final_file,sheet_name='Feedback',index=False)       
    return outdf

def Extract_Feedback_Concept(ffile,cfiles):
    
    xl = pd.ExcelFile(ffile)
    feedback = xl.parse('National Issue')
    feedback['SNO'].replace('', np.nan, inplace=True)
    feedback.dropna(subset=['SNO'], inplace=True)
    for index, row in feedback.iterrows():
        strg=str(row['SNO'])
        strg=re.sub("[^0-9]", "",strg)
        feedback.at[index,'SNO']=strg
    feedback.drop(['Category'],inplace=True,axis=1)    
    feedback['SNO'] = feedback['SNO'].astype('int32')
    print(str(len(feedback))+' rows read into Feedbacks dataframe from file '+feedback_file)
    
    df3=pd.DataFrame()
    for concept_file in cfiles:
        concept_file=concept_file.replace('\n','')
        cxl = pd.ExcelFile(path+concept_file)
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
    print(str(len(concepts))+' concepts extracted from concept dataframe built from concept files ')
 
    return feedback, concepts

def Merge_Feedback_Concepts_Unmatched(feedback,concepts,final_df):
    
    df3=pd.merge(feedback,concepts[['SNO','Concept']],on='SNO',how='left')
    print('Number of rows after left outer join Feedback files and Concepts files on SNO:',len(df3))
   
    for index, row in df3.iterrows():
        strg=row['Feedback']
        strg=strg.replace("\n"," ")
        df3.at[index,'Feedback']=strg
   
    df3['Label'] = df3['Label'].str.lower()
    df3=df3[df3['Concept'].isnull()]
    df3.drop(['Concept'],inplace=True,axis=1)
    df3=df3.groupby(['Label'],as_index=False).agg(lambda x:x.values.tolist())
    print('Number of rows after aggregating on Label:',len(df3))

    return df3, len(df3)

def Merge_Feedback_Concepts_Matched(feedback,concepts,final_df):
    
    df3=pd.merge(feedback,concepts[['SNO','Concept']],on='SNO',how='inner')
    print('Number of rows after inner joining Feedback files and Concepts files on SNO:',len(df3))
    
    final_df,nrows=Extractive_Summarize(df3,final_df)

    return final_df, nrows
        
def Extractive_Summarize(df3,final_df):
    
    for index, row in df3.iterrows():
            strg=row['Feedback']
            strg=strg.replace("\n"," ")
            df3.at[index,'Feedback']=strg
    df3['Label'] = df3['Label'].str.lower()
    df3=df3.groupby(['Label','Concept'],as_index=False).agg(lambda x:x.values.tolist())
    
    for row in df3.index:
        if len(df3.at[row,'Feedback']) > 1:
           for idx, feed in enumerate(df3.at[row,'Feedback']):
               if not feed.endswith('.'):
                  df3.at[row,'Feedback'][idx] = df3.at[row,'Feedback'][idx]+'.'
                  
           fbk=' '.join(map(str,df3.at[row,'Feedback']))
           summ=summarize(fbk,word_count=400)
           if not summ:
              summ=fbk
        else:
           summ=' '.join(map(str,df3.at[row,'Feedback']))
        df3.at[row,'Feedback']=summ.replace("\n"," ")   
    print('Number of rows after aggregating on Label and Concept:',len(df3))
 
    return df3, len(df3)

def Merge_Public_PA_data(final_df_matched_summarized,final_df_unmatched_summarized):
    
    #pa=open(pafile,'r')
    turl=open('/users/pivotalit/downloads/cnn-dailymail/url_lists/all_train_copy.txt','a')
    vurl=open('/users/pivotalit/downloads/cnn-dailymail/url_lists/all_val_copy.txt','a')
    eurl=open('/users/pivotalit/downloads/cnn-dailymail/url_lists/all_test_copy.txt','w')

    #fdlst=pa.read().split('\n')
    
    train_set=random.sample(range(len(final_df_matched_summarized)-1),round((len(final_df_matched_summarized)-1)*.95))
    val_set=list(set(range(len(final_df_matched_summarized)-1))-set(train_set))
    #val_set=random.sample(test_set,round(len(test_set)*.5))
    #test_set=list(set(test_set)-set(val_set))

    for i in train_set:
        line=final_df_matched_summarized.loc[i]
        fname='pa_train_'+str(i)+'.story'
        turl.write('\n'+'pa_train_'+str(i))
        ofile=open('/users/pivotalit/downloads/cnn/stories/'+fname,'w')
        ofile.write(line[3]+'\n\n')
        ofile.write('@highlight'+'\n')
        ofile.write(line[1])
        ofile.close()
    
    for i in val_set:
        line=final_df_matched_summarized.loc[i]
        fname='pa_val_'+str(i)+'.story'
        vurl.write('\n'+'pa_val_'+str(i))
        ofile=open('/users/pivotalit/downloads/cnn/stories/'+fname,'w')
        ofile.write(line[3]+'\n\n')
        ofile.write('@highlight'+'\n')
        ofile.write(line[1])
        ofile.close()

    for index,row in final_df_unmatched_summarized.iterrows():  
        line=final_df_unmatched_summarized.loc[index]
        fname='pa_test_U_'+str(index)+'.story'
        eurl.write('pa_test_U_'+str(index)+'\n')
        ofile=open('/users/pivotalit/downloads/cnn/stories/'+fname,'w')
        ofile.write(line[3]+'\n\n')
        ofile.write('@highlight'+'\n')
        ofile.write(line[1])
        ofile.close()        

        
if __name__ == '__main__':
   matched_summarized=path+'matched_feedback_concepts.tsv'
   unmatched_summarized=path+'unmatched_clustered_labels.tsv'
   mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
   if os.path.exists(matched_summarized):
      os.remove(matched_summarized)
   if os.path.exists(unmatched_summarized):
      os.remove(unmatched_summarized)
   
   f=open(path+'input_files.tsv','r')
   files=f.readlines()
   f.close()
   final_df_matched_summarized = pd.DataFrame()
   final_df_unmatched = pd.DataFrame()
   final_df_unmatched_summarized = pd.DataFrame()
   feedback=pd.DataFrame()
   concepts=pd.DataFrame()
   
   print('####### Extracting Feedback text and concepts from excel files #######')
   for file in files:
       feedback_file=file.split('\t')[0]
       concept_files=file.split('\t')[1].split(',')      
       fb, cp = Extract_Feedback_Concept(path+feedback_file,concept_files)
       feedback=feedback.append(fb,ignore_index=True)
       concepts=concepts.append(cp,ignore_index=True)
   print('-----------------------------------------------------------------------------')
   print('Total '+str(len(feedback))+' feedback text and '+str(len(concepts))+' concepts')
   print('-----------------------------------------------------------------------------')
  
   print('###### Pulling all feedback text that DON''T have concepts generated ########')
   final_df_unmatched, nrows=Merge_Feedback_Concepts_Unmatched(feedback,concepts,final_df_unmatched)

   print('###### Pulling all feedback text that have concepts ########')
   final_df_matched_summarized, nrows=Merge_Feedback_Concepts_Matched(feedback,concepts,final_df_matched_summarized)
   saveplot=False
   print('--------------------------------------------------------------------')
   print('Clustering started for feedback text under each Label that DON''T have concepts......')
   outdf=Cluster_Unmatched_Feedback(final_df_unmatched,saveplot)
   final_df_unmatched_summarized, nrows = Extractive_Summarize(outdf,final_df_unmatched_summarized)
   print('Concepts to be generated from '+str(nrows)+' Labels ')
   print('--------------------------------------------------------------------')

#   final_df_matched_summarized=pd.read_csv(matched_summarized,sep='\t',header=None) 
#   final_df_unmatched_summarized=pd.read_csv(unmatched_summarized,sep='\t',header=None) 
   
   shutil.copyfile('/users/pivotalit/downloads/cnn-dailymail/url_lists/all_train.txt','/users/pivotalit/downloads/cnn-dailymail/url_lists/all_train_copy.txt')
   shutil.copyfile('/users/pivotalit/downloads/cnn-dailymail/url_lists/all_val.txt','/users/pivotalit/downloads/cnn-dailymail/url_lists/all_val_copy.txt')
   shutil.copyfile('/users/pivotalit/downloads/cnn-dailymail/url_lists/all_test.txt','/users/pivotalit/downloads/cnn-dailymail/url_lists/all_test_copy.txt')
   
   fileList = glob.glob('/users/pivotalit/downloads/cnn/stories/pa_*')
   for filePath in fileList:
       os.remove(filePath)
   Merge_Public_PA_data(final_df_matched_summarized,final_df_unmatched_summarized)
# 
   final_df_matched_summarized.to_csv(matched_summarized,sep='\t',header=False)
   final_df_unmatched_summarized.to_csv(unmatched_summarized,sep='\t',header=False)

   