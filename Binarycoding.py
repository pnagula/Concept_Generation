import time
import csv
from xlsxwriter.workbook import Workbook
import pandas as pd
import sys
import os 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

path=sys.argv[1]

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score['pos'],score['neg'],score['neu'],score['compound']
    
def remove_punct(sno):
    sno=sno.replace('[','')
    sno=sno.replace(']','')
    sno=sno.replace(' ','')
    return sno.split(',')

if __name__ == '__main__':
    analyser = SentimentIntensityAnalyzer()
    timestr = time.strftime("%Y%m%d")
    xl = pd.ExcelFile(path+'/files/output_files/Important_Concepts.xlsx')
    fdf = xl.parse('Sheet1')
    fdf=fdf.groupby(['Label'],as_index=False).agg(lambda x:x.values.tolist())
    fdf.drop('Index',axis=1,inplace=True)
    for index,row in fdf.iterrows():
        cps='\t'.join(row['Concept'])
        fdf.loc[index,'Concept']=cps
    
    concepts=[[cp for cp in conc.split("\t")] for conc in fdf['Concept'] ]
    concepts=[cp1.strip() for cp in concepts for cp1 in cp]
    unq_concepts=dict.fromkeys(concepts)
    unq_concepts_list=list(dict.fromkeys(concepts))
    bfdf=pd.DataFrame(columns=['SNO','Issue','Score','Sentiment_Positive','Sentiment_Negative','Sentiment_Neutral','Sentiment_Compound'])
    bfdf=pd.concat([bfdf,pd.DataFrame(columns=unq_concepts)],sort=False)
    bfdf.to_csv(path+'/Binary_Coded_'+timestr+'.tsv',header=True,index=False,sep='\t')
    fp=open(path+'/Binary_Coded_'+timestr+'.tsv','a')
    for index, row in fdf.iterrows():
        concepts=str(row['Concept']).split("\t")
        for ix,sn in enumerate(row['SNO']):
            pos,neg,neu,comp=sentiment_analyzer_scores(concepts[ix].strip())
            strg=str(sn)+'\t'+row['Label']+'\t'+str(row['Score'][ix])+'\t'+str(pos)+'\t'+str(neg)+'\t'+str(neu)+'\t'+str(comp)+'\t'
            sequence=(('0,')*len(unq_concepts)).split(',')
            sequence=sequence[:-1]
            sequence[unq_concepts_list.index(concepts[ix].strip())]='1'
            sequence='\t'.join(sequence)
            strg=strg+sequence+'\n'
            fp.write(strg)
    fp.close()

    ## Convert .tsv to xlsx file
    # Add some command-line logic to read the file names.
    tsv_file = path+'/files/output_files/Binary_Coded_'+timestr+'.tsv'
    xlsx_file = path+'/files/output_files/Binary_Coded_'+timestr+'.xlsx'
   
    if os.path.exists(xlsx_file):
       os.remove(xlsx_file)
    # Create an XlsxWriter workbook object and add a worksheet.
    workbook = Workbook(xlsx_file)
    worksheet = workbook.add_worksheet()

    # Create a TSV file reader.
    tsv_reader = csv.reader(open(tsv_file, 'r'), delimiter='\t')

    # Read the row data from the TSV file and write it to the XLSX file.
    for row, data in enumerate(tsv_reader):
        worksheet.write_row(row, 0, data)

    # Close the XLSX file.
    workbook.close()
 
    # remove .tsv file
    if os.path.exists(path+'/files/output_files/Binary_Coded_'+timestr+'.tsv'):
       os.remove(path+'/files/output_files/Binary_Coded_'+timestr+'.tsv')
