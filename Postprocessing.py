#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:25:59 2019

@author: pnagula
"""

from rake_nltk import Rake
from nltk.tokenize import sent_tokenize,word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import os
import glob
from nltk.corpus import stopwords
import numpy as np
import spacy 
import string
from fuzzywuzzy import fuzz,process
from gensim.utils import lemmatize
from spellchecker import SpellChecker
import networkx as nx
import community
import random
import matplotlib.pyplot as plt
import webcolors

path=sys.argv[1]
 
def Rank_Concepts(cs,SNO):
    simdf=pd.DataFrame(columns=['Cptples','Score'])
    maxsim=pd.DataFrame(columns=['Src_Idx','Dest_Idx','Source','Destination','Score'])
    for i in range(cs.shape[0]):
        for j in range(cs.shape[1]):
            if i==j:
               cs[i,j]=0.0
    D=nx.Graph()
    for i in range(cs.shape[0]): 
        if np.max(cs[i,:])>=np.median(cs):
               simdf=simdf.append({'Cptples':(i,np.argmax(cs[i,:])),'Score':np.max(cs[i,:])},ignore_index=True)
               maxsim=maxsim.append({'Src_Idx':i,'Dest_Idx':np.argmax(cs[i,:]),'Source':SNO[i],'Destination':SNO[np.argmax(cs[i,:])],'Score':np.max(cs[i,:])},ignore_index=True)
               D.add_weighted_edges_from([(i,np.argmax(cs[i,:]),np.max(cs[i,:]))]) 
    simdf.set_index(['Cptples'],drop=True,inplace=True)
    return simdf,maxsim,community.best_partition(D),D
   
def Remove_duplicates(final_df,simthreshold):
    outdf=pd.DataFrame(columns = ['SNO','Label','Concept'])
    spell = SpellChecker(distance=1)
    swords=list(stopwords.words('english'))
    swords.append('be')
    for index,row in final_df.iterrows():
        vect=np.zeros(300)
        cplist=row['Concept']
        snolist=row['SNO']
        for cp in cplist:
            cp=" ".join([spell.correction(word) if word not in string.punctuation else word for word in word_tokenize(cp)])
            lemmatized_sent=" ".join([wd.decode('utf-8').split('/')[0] for wd in lemmatize(cp,stopwords=swords)])
            doc=nlp(lemmatized_sent) 
            if doc.vector_norm:
               vect=np.vstack((vect,doc.vector/doc.vector_norm))
            else:
               vect=np.vstack((vect,doc.vector)) 
        cs=cosine_similarity(vect[1:,])
        delete=[]
        for i in range(cs.shape[0]):
            for j in range(cs.shape[1]):
                if (cs[i,j]>simthreshold and i!=j):
                   if len(final_df.at[index,'Concept'][i]) < len(final_df.at[index,'Concept'][j]):
                         delete.append(i)
                   else:
                         delete.append(j) 
        delete=list(dict.fromkeys(delete))
        cpdict={}
        snodict={}
        for ix,element in enumerate(zip(cplist,snolist)):
            cpdict[ix]=element[0]
            snodict[ix]=element[1]
        for key in delete:
            del cpdict[key]
            del snodict[key]
        
        for cpt,sno in zip(list(cpdict.values()),list(snodict.values())):
            outdf=outdf.append({'SNO':sno,'Label':row['Label'],'Concept':cpt},ignore_index=True)
    return outdf    

def Remove_TitlesinConcept(final_df,labelmatch):
    for index,row in final_df.iterrows():
        concept=row['Concept']
        acronyms=[word for word in word_tokenize(row['Feed']) if word.isupper() and word!='A']
        acronyms=list(dict.fromkeys(acronyms)) 
        
        label=row['Label']
        label=label.lower()
        new_label=' '
        for ch in label:
            if ch in string.punctuation:
               new_label=new_label+' '+ch+' '
            else:
               new_label=new_label+ch
        label=new_label
        
        new_concept=' '
        for ch in concept:
            if ch in string.punctuation:
               new_concept=new_concept+' '+ch+' '
            else:
               new_concept=new_concept+ch
        concept=new_concept
        concept=concept.lower()
        concept=word_tokenize(concept)
        matches= process.extract(label, concept,scorer=fuzz.token_set_ratio,limit=3)
        label=word_tokenize(label)
        ixlist=[]
        for match in matches:
          if match[1]>=90:
             ixlist.append(label.index(match[0]))
        if ixlist:
            match=matches[np.argmin(ixlist)]
            start_cp = concept.index(match[0])
            start_lb = label.index(match[0])
            end_lb=len(label)
            if start_cp+len(label[start_lb:end_lb]) > len(concept):
                   end_cp=len(concept)
            else:
                   end_cp=start_cp+len(label[start_lb:end_lb])
            mscore= fuzz.token_sort_ratio(' '.join(label),' '.join(concept[start_cp:end_cp]))
            if mscore>=labelmatch:
               exclude=concept[start_cp:end_cp]
               strg=' '.join([cp for cp in concept if cp not in exclude])
               row['Concept']=strg
        #replace acronyms with their uppercase version       
        ntok=[]    
        for tok in word_tokenize(row['Concept']):
            if len(acronyms)>0:
               if tok.upper() in acronyms:
                  ntok.append(tok.upper())
               else:
                  ntok.append(tok)
            else:
               ntok.append(tok)
        if len(ntok) < 4:
           final_df.at[index,'Concept']=None
        else:   
           final_df.at[index,'Concept']= ' '.join(ntok)   
                
    final_df.drop(['Feed'],axis=1,inplace=True)
    final_df.dropna(subset=['Concept'] , inplace=True)
    
    return final_df       
    
if __name__ == '__main__':
 
    #read postprocessing parameters
    print('####### Reading Postprocessing parameters ################')
    parmfile=open(path+'/files/static_data/postprocessing_parameters.txt')
    parms=[]
    for line in parmfile:
        if line!='\n':
           if line.find('=')==-1:
              raise ValueError('Parameter should be either simthreshold=<value> or max_concepts=<value> or labelmatch=<value> or minimum_feedbacks=<value>') 
           else:
              parmname=line[0:line.index('=')]
              value=line[line.index('=')+1:len(line)]
              if not value:
                 raise ValueError('Provide some value to the parameter')
              else:   
                 if  parmname!='simthreshold' and parmname!='max_concepts' and parmname!='labelmatch' and parmname!='minimum_feedbacks':
                     raise ValueError('Parameter name can only be simthreshold, max_concepts, labelmatch, minimum_feedbacks')
                 else:
                     parms.append(value) 
    if parms[0]:
       simthreshold=float(parms[0])
       print('Duplicate removal Similarity Threshold:',simthreshold)
    if parms[1]:
       max_concepts=int(parms[1])
       print('Maximum number of Concepts per Issue:',max_concepts)
    if parms[2]:
       labelmatch=int(parms[2]) 
       print('Label removal match Threshold:',labelmatch)
    if parms[3]:
       minimum_feedbacks=int(parms[3])
       print('Minimum number of Feedbacks per Issue:',minimum_feedbacks)

    nlp = spacy.load("en_vectors_web_lg")
    folders=['decode_test_400maxenc_4beam_35mindec_100maxdec_ckpt-153886']
    namesp=open(path+'/files/static_data/namesdb.pickle', 'rb')
    namesdict=pickle.load(namesp)
    df=pd.DataFrame(columns=['Index','Concept'])
    for folder in folders:
        fileList = glob.glob(path+'/files/pglogs/firstpgexp_400/'+folder+'/decoded/*_decoded.txt')
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
    df=df.groupby(['Index'],as_index=False).agg(lambda x:''.join(x.values))
    file=path+'/files/output_files/unmatched_labels.tsv'
    if os.path.exists(file)==False:
       raise ValueError('File unmatched_labels.tsv not found.') 
    unmatched=pd.read_csv(file,sep='\t',header=None) 
    unmatched.columns=['Index','SNO','Feedback','Label']       
    unmatched=pd.merge(unmatched,df,how='inner',on='Index')
    if os.path.exists(path+"/files/output_files/consolidated_concepts.tsv"):
       os.remove(path+"/files/output_files/consolidated_concepts.tsv")
    unmatched.to_csv(path+'/files/output_files/consolidated_concepts.tsv',sep='\t',index=False)    

    final_df=pd.DataFrame(columns=['SNO','Label','Feed','Concept'])
    print('######### Removing Names,Omitting pronouns and extract concepts from summary ########')
    fbcnt=unmatched.groupby('Label',as_index=False).count()
    fbcnt.drop(['SNO','Feedback','Concept'],axis=1,inplace=True)
    fbcnt.columns=['Label','Count']
    unmatched=pd.merge(unmatched,fbcnt,how='inner',on='Label')
    for index,row in unmatched.iterrows():
       if row['Count']>=minimum_feedbacks:
            text=row['Concept']
            summ=row['Feedback']
            label=row['Label']
            
            text=text.replace(' .','. ')
            text=text.replace('-lrb-','')
            text=text.replace('-rrb-','')
            
            # remove words in omit list from summary
            omitlist=['[unk]','respondent','unsolicited','solicited','he','she','ms','miss','mrs','mr','mdm','madam']
            text1=' '.join([tok for tok in word_tokenize(text) if tok.lower() not in namesdict and tok.lower() not in omitlist])
            
            # extract relevant content
            punct='.?!' # sentence breakers
            #punct='.;?!|/#(){}[]'
            r = Rake()
            sents=sent_tokenize(text1)
            r.extract_keywords_from_sentences(sents)
            phrases=r.get_ranked_phrases()
            fsent=[]
            for phrase in phrases:
                for sent in sents:
                    if phrase in sent.lower():
                       tsent=sent[0:sent.lower().index(phrase)]
                       sidx=0
                       for t in reversed(tsent):
                           if t in punct:
                              s="".join(reversed(tsent))
                              sidx=len(tsent)-s.index(t)
                              break 
                       tsent=sent[sent.lower().index(phrase):]
                       eidx=len(sent)
                       for t in tsent:
                           if t in punct:
                              eidx=sent.lower().index(phrase)+tsent.index(t)
                              break
                       fsent.append(sent[sidx:eidx].strip()) 
                          
            fsent=list(dict.fromkeys(fsent)) 
            fsent=[sent for sent in fsent if len(word_tokenize(sent))>3]
            if not fsent == []:
               fsent=dict(enumerate(fsent))
               fsent=[sent for sent in fsent.values()]
               cpts=pd.DataFrame(fsent)
               cpts.columns=['Concept']
               cpts['Label']=row['Label']
               cpts['SNO']=row['SNO']
               cpts['Feed']=summ
               cpts=cpts[['SNO','Label','Feed','Concept']]
               final_df=final_df.append(cpts)
    final_df.reset_index(drop=True,inplace = True) 
    
    # remove issue/labels text from concept
    print('########## Removing issue/label text from concept ############')
    final_df=Remove_TitlesinConcept(final_df,labelmatch) 
 
    # remove duplicate sentences
    print('########## Removing duplicate concepts ####################')
    final_df=final_df.groupby(['Label'],as_index=False).agg(lambda x:x.values.tolist()) 
    final_df=Remove_duplicates(final_df,simthreshold)
    
    # Community Detection and Pagerank concepts and select unique most important concepts from each community
    print('########## Clustering and Scoring Concepts ####################')
    final_df=final_df.groupby(['Label'],as_index=False).agg(lambda x:x.values.tolist()) 
    important_concepts=pd.DataFrame(columns = ['Label','SNO','Concept','Score'])
    spell = SpellChecker(distance=1)
    swords=list(stopwords.words('english'))
    swords.append('be')
    for index,row in final_df.iterrows():
        cplist=row['Concept']
        vect=np.zeros(300)
        for cp in cplist:
            cp=" ".join([spell.correction(word) if word not in string.punctuation else word for word in word_tokenize(cp)])
            lemmatized_sent=" ".join([wd.decode('utf-8').split('/')[0] for wd in lemmatize(cp,stopwords=swords)])
            doc=nlp(lemmatized_sent) 
            if doc.vector_norm:
               vect=np.vstack((vect,doc.vector/doc.vector_norm))
            else:
               vect=np.vstack((vect,doc.vector)) 
        cs=cosine_similarity(vect[1:,])
        simdf,maxsim,partition,G=Rank_Concepts(cs,row['SNO']) 
        maxsim=maxsim.groupby(['Dest_Idx'],as_index=False).agg(lambda x:x.values.tolist())
        for ix,rw in maxsim.iterrows():
            maxsim.at[ix,'Count']=len(maxsim.at[ix,'Score'])
        part=pd.DataFrame(list(partition.items()),columns=['Node','Community'])
        part1=part.groupby(['Community'],as_index=False).count()
        part1.columns=(['Community','Count'])
        part=pd.merge(part,part1,how='inner',on='Community')
        part=part.groupby(['Community'],as_index=False).agg(lambda x:x.values.tolist()) 
        part.sort_values(['Count'],ascending=False,inplace=True)
        cnt=0
        maxscore=0.0
        outdf=pd.DataFrame(columns = ['Label','SNO','Concept','Score'])
        for index,rw in part.iterrows():
            if cnt < max_concepts:
               list_nodes=rw['Node'] 
               D=nx.DiGraph()
               for i in list_nodes:
                   for j in list_nodes:
                       if i!=j and (i,j) in simdf.index:
                          D.add_weighted_edges_from([(i,j,cs[i,j])]) 
               pgranks=pd.DataFrame(list(nx.pagerank(D).items()),columns=['Node','Pagerank'])           
               pgranks.sort_values(['Pagerank'],ascending=False,inplace=True)
               score=rw['Count'][0]+pgranks['Pagerank'].iloc[0]
               node=pgranks['Node'].iloc[0]
               if score>maxscore:
                  maxscore=score
               outdf=outdf.append({'Label':row['Label'],'SNO':row['SNO'][node],'Concept':row['Concept'][node],'Score':score},ignore_index=True)      
               cnt+=1
        for index,rw in outdf.iterrows():
            important_concepts=important_concepts.append({'Label':row['Label'],'SNO':rw['SNO'],'Concept':rw['Concept'],'Score':rw['Score']/maxscore},ignore_index=True)
    important_concepts.sort_values(['Label','Score'],ascending=False,inplace=True)
    important_concepts.to_excel(path+'/files/output_files/Important_Concepts.xlsx',index_label='Index')
